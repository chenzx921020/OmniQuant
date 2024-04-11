import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state,\
                            smooth_and_quant_temporary_quantrainer,smooth_and_quant_inplace_quantrainer
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")
import torch.nn.functional as F
from transformers import Trainer
import transformers
from datautils import get_wikitext_for_trainer
from torch.utils.checkpoint import checkpoint
from utils import ampscaler_get_grad_norm
from torch.optim import lr_scheduler
#torch.autograd.set_detect_anomaly(True)

def kl_loss(output, target, temperature):
    output = F.log_softmax(output / temperature, dim=-1)
    target = F.softmax(target / temperature, dim=-1)
    return F.kl_div(output, target, reduction="batchmean")

def fp32_16_pre_hook(module, input):
    i = input[0].to(torch.bfloat16) #half()
    return (i,)


def fp16_32_hook(module, _, output):
    return output.float()
    
class QuantKDTrainer(Trainer):
    #def compute_loss(self, model, inputs):
    def training_step(self, model, inputs):
        #import pdb;pdb.set_trace()
        hooks_to_remove = []
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
            #if isinstance(module, nn.Linear):
                module = module.to(torch.bfloat16) #half()
                h1 = module.register_forward_pre_hook(fp32_16_pre_hook)
                h2 = module.register_forward_hook(fp16_32_hook)
                hooks_to_remove.append(h1)
                hooks_to_remove.append(h2)
                module.use_temporary_parameter=False
        
        #import pdb;pdb.set_trace()
        with torch.no_grad():
            raw = model(**inputs)
        #layers = model.model.layers
        #import pdb;pdb.set_trace()
        for i in range(len(model.model.layers)):
            #import pdb;pdb.set_trace()
            layer = model.model.layers[i]
            layer = layer.to(torch.bfloat16) # half()
            # with torch.cuda.amp.autocast():
            checkpoint(smooth_and_quant_temporary_quantrainer,layer,use_reentrant=False)
            torch.cuda.empty_cache()
        #with torch.cuda.amp.autocast():
        outputs = model(**inputs)
        loss = kl_loss(outputs.logits, raw.logits.detach(), 1)
        
        #import pdb;pdb.set_trace()    
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if labels is not None:
            loss += self.label_smoother(outputs, labels, shift_labels=True)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # choose loss
            loss += outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        #print(f"global loss is: ",loss)
        # loss_scaler = utils.NativeScalerWithGradNormCount()
        # norm = loss_scaler(loss, self.optimizer,parameters= get_omni_parameters(model, True)).cpu()
        #loss = loss.to(torch.bfloat16)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        #import pdb;pdb.set_trace()
        self.optimizer.step()
        self.optimizer.zero_grad()
        for hook in hooks_to_remove:
            hook.remove()
        hooks_to_remove.clear()
        return loss.detach()

def omniquant_global_v3(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("... Starting global finetune ...")
    model = lm.model
    dev = lm.device
    #dev = torch.device('cpu')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    optimizer = torch.optim.AdamW(
        [{"params":let_parameters(layers, True),"lr":args.let_lr}, {"params":lwc_parameters(layers),"lr":args.lwc_lr}],weight_decay=args.wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.85) 
    if args.epochs > 0:    
        dataset=get_wikitext_for_trainer(lm.tokenizer,seqlen=1024)
        # Freezing the original weights
        for name,param in model.named_parameters():
            if "smooth_scale" in name or "smooth_shift" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # if param.ndim ==1:
        #     param.data = param.data.to(torch.float32)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        #pdb.set_trace()
        # Training
        trainer =  QuantKDTrainer(
            model=model,
            train_dataset=dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=100,
                num_train_epochs= args.epochs,
                #learning_rate=1e-6,
                bf16=True,
                logging_steps=100,
                output_dir='outputs',
                save_steps=-1,
                #weight_decay=1e-5,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(lm.tokenizer, mlm=False),
            optimizers = (optimizer,scheduler),
            
        )

        model.config.use_cache = False
        trainer.train()

    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        clear_temp_variable(layer)
        register_scales_and_zeros(layer)
        smooth_and_quant_inplace_quantrainer(layer)
        #layer=layer.to(torch.float32)
        layer=layer.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()   
    model.to(torch.float16)

    return model