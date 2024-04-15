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
import scipy.stats
#torch.autograd.set_detect_anomaly(True)

def kl_loss(output, target, temperature):
    outputs = F.log_softmax(output / temperature, dim=-1)
    targets = F.softmax(target / temperature, dim=-1)
    return F.kl_div(outputs, targets, reduction="batchmean")

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
        loss = kl_loss(outputs.logits, raw.logits.detach(), 80)
        
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

        # loss_scaler = utils.NativeScalerWithGradNormCount()
        # norm = loss_scaler(loss, self.optimizer,parameters= get_omni_parameters(model, True)).cpu()
        #loss = loss.to(torch.bfloat16)
        #print(f"global loss is: ", loss)
        loss.backward()
        #import pdb;pdb.set_trace()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        #import pdb;pdb.set_trace()
        self.optimizer.step()
        self.optimizer.zero_grad()
        for hook in hooks_to_remove:
            hook.remove()
        hooks_to_remove.clear()
        return loss.detach()

def omniquant_global_v2(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    #dev = torch.device('cpu')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    #pdb.set_trace()
    #layers[0] = layers[0].module
    #model = model.module
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    #import pdb;pdb.set_trace()
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    
        #dataset=get_wikitext_for_trainer(tokenizer,seqlen=1024)
    for i in range(len(layers)):
        logger.info(f"=== Start process layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            #print ('pairs key: ',pairs[key])
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
        clear_temp_variable(qlayer)
        #register_scales_and_zeros(qlayer)
        layers[i]=qlayer
        #qlayer.to('cpu')
        del layer
    layers = model.model.layers
        #del layer
    torch.cuda.empty_cache()
    #pdb.set_trace()
    optimizer = torch.optim.AdamW(
        [{"params":let_parameters(layers, True),"lr":args.let_lr}, {"params":lwc_parameters(layers),"lr":args.lwc_lr}],weight_decay=args.wd)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.85) 
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-6)
    #import pdb;pdb.set_trace()
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
                label_smoothing_factor=0.1,
                #max_steps=1924,
                #learning_rate=1e-6,
                bf16=True,
                logging_steps=50,
                output_dir='outputs',
                save_steps=-1,
                #weight_decay=1e-5,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(lm.tokenizer, mlm=False),
            # optimizers = (optimizer,scheduler),
            optimizers = (optimizer,scheduler),
        )

        model.config.use_cache = False
        trainer.train()
    #pdb.set_trace()
    
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
