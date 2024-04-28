import torch,os,sys,torch.nn as nn,torch.nn.functional as F,easydict
sys.path.append("../")
from transformers import AutoTokenizer,AutoModelForCausalLM
from quantize.quantizer import UniformAffineQuantizer
from quantize.utils import smooth_ln_fcs_inplace,smooth_fc_fc_inplace,smooth_q_k_inplace,\
                            set_quant_state,register_scales_and_zeros,smooth_and_quant_inplace_quantrainer
from easydict import EasyDict
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import datasets,torch,torch.nn as nn,tqdm,random,transformers
from datasets import load_dataset
from transformers import AutoTokenizer
from hx.eval_ppl import eval_ppl
from lm_eval import evaluator
from tqdm import tqdm
from models.int_llama_layer import QuantLlamaDecoderLayer
import utils
from models.LMClass import LMClass
import argparse
import gc
import numpy as np
from datautils import get_loaders

def chat(model,tokenizer,question="who are u?",max_length=128):
    instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible,.
            \n<</SYS>>\n\n{} [/INST]"""
    format_input = instruction.format(question)
    token = tokenizer(format_input, return_tensors="pt")
    output = model.generate(token['input_ids'].to(model.device), do_sample=True, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print (response)
    return response

@torch.no_grad()
def evaluate(lm, args,dev):
    results = {}
    if args.multigpu:
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
            lm.model = lm.model.to(dev)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)


    if args.eval_ppl:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2","c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                #logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(dev)
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            #logger.info(f'{dataset} : {ppl.item()}')
            print ('ppl:',ppl.item())
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.weight_quant_params = {
    "n_bits": 4,
    "per_channel_axes": [0],
    "symmetric": False,
    "dynamic_method": 'per_channel',
    "group_size": None,
    "lwc":True,
    "disable_zero_point": False
}
args.act_quant_params = {
    "n_bits":  8,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.q_quant_params = {
    "n_bits": 8,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.k_quant_params = {
    "n_bits": 8,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.v_quant_params = {
    "n_bits": 8,
    "per_channel_axes": [],
    "symmetric": False,
    "dynamic_method": 'per_token',
}
args.p_quant_params = {
    "n_bits": 16,
    "metric": "fix0to1",
}
args.attn_implementation='eager'
args.model = "/data01/ssd/llama2-7b-chat-hf"
args.batch_size = 1
args.multigpu=False
args.net='Llama-2-7b-chat'
args.eval_ppl=True
args.cache_dir='./cache'
args.model_family = args.net.split('-')[0]
args.limit = -1
args.seed =2
torch.set_grad_enabled(False)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

lm = LMClass(args)
dev = torch.device('cuda:2')
lm.seqlen = 2048
lm.model.eval()
for param in lm.model.parameters():
    param.requires_grad = False

ckp = "./log/llama-7b-w4a8-global_ft_20240424/omni_parameters_global_ft.pth"
st = torch.load(ckp,map_location="cuda:2")
print(list(st[0].keys()))

DecoderLayer = QuantLlamaDecoderLayer

if True:
    layers = lm.model.model.layers
    
    for i in range(len(layers)):
        qlayer = DecoderLayer(lm.model.config, layers[i], args).to(dev)
        #qlayer.float()
        layer_params = EasyDict(st[i])
        # layer = model.model.layers[i].cuda()
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  
        #clear_temp_variable(qlayer)
        #smooth_and_quant_inplace_quantrainer(qlayer)
        smooth_ln_fcs_inplace(qlayer.input_layernorm,[qlayer.self_attn.q_proj, qlayer.self_attn.k_proj, qlayer.self_attn.v_proj],
                                layer_params.qkv_smooth_scale.half(),layer_params.qkv_smooth_shift.half())
        smooth_ln_fcs_inplace(qlayer.post_attention_layernorm,[qlayer.mlp.up_proj,qlayer.mlp.gate_proj],
                                layer_params.fc1_smooth_scale.half(),layer_params.fc1_smooth_shift.half())
        smooth_fc_fc_inplace(qlayer.self_attn.v_proj,qlayer.self_attn.o_proj,
                            layer_params.out_smooth_scale.half(), layer_params.out_smooth_shift.half())
        smooth_q_k_inplace(qlayer.self_attn.q_proj, qlayer.self_attn.k_proj,
                        layer_params.qkt_smooth_scale.half())
        #smooth_fc_fc_inplace(qlayer.mlp.up_proj,qlayer.mlp.down_proj,layer_params.fc2_smooth_scale,None) # 进行up & down的平衡
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                weight_quantizer = UniformAffineQuantizer(**args.weight_quant_params,shape=module.weight.shape).to(dev) #cuda()
                weight_quantizer.upbound_factor.data.copy_(layer_params.get(".".join([name,"weight_quantizer","upbound_factor"])).half())
                weight_quantizer.lowbound_factor.data.copy_(layer_params.get(".".join([name,"weight_quantizer","lowbound_factor"])).half())
                module.weight.data = weight_quantizer(module.weight)
                module.use_temporary_parameter=False
        register_scales_and_zeros(qlayer)
        qlayer.half()
        layers[i] = qlayer.cpu()
        del qlayer
        torch.cuda.empty_cache()
    #lm.model.model.layers=layers
    torch.cuda.empty_cache()
    gc.collect()

#evaluate(lm, args, dev)
lm.model.to(dev)

#chat(lm.model,lm.tokenizer,max_length=128)


questions = [
    """请仿照白居易的风格写一首诗。""",
    """晚上睡不着应该怎么办。""",
    """对下面文本纠错：下个星期，我跟我朋唷打算去法国玩儿。""",
    """问题：拥有大麻是否合法？
选项：
A.这取决于你所在的国家。
B.是的，拥有大麻是合法的。
C.不，拥有大麻是非法的。""",
    """以下句子翻译为英文：
坎儿井（Karez）是新疆干旱地区的一种水利系统,由地下渠道将水井连接而成。
该系统将春夏季节渗入地下的大量雨水及积雪融水收集起来,通过山体的自然坡度引到地面,用于灌溉农田和满足人们的日常用水需求。
坎儿井减少了水在地面的蒸发,对地表破坏很小,因而有效地保护了自然资源与生态环境。""",
    """Translate the following sentences into Chinese:
The Karez(坎儿井) is a water conservancy system in the arid regions of Xinjiang, consisting of underground channels connecting Wells. The system collects a large amount of rain and meltwater of snow that seeps into the ground in spring and summer and leads it to the ground through the natural slope of the mountain.
The water is brought to the ground and used to irrigate farms and meet people's daily water needs. The Karez reduces evaporation of water at the surface, does little damage to the surface, and thus effectively protects the natural resources and ecological environment.""",
    """计算123*456的结果。""",
    """计算123*456789的结果。""",
    """曲线𝑦 = (x-1)^2 在点(-1,4)处的切线方程为( )"""
    # """请生成一道和“数组”与“合并”概念有关的编程任务题目，并给出一段 Python 函数来解决此任务。""",
    # """问题描述：检查给定的数字列表中是否存在两个数字的距离小于给定阈值。请给出一段 Python 代码来解决此任务。""",
    # """问题描述：检查给定的数字列表中是否存在两个数字的距离小于给定阈值。请给出一段 Python 代码来解决此任务。""",
    '''冯诺依曼架构是什么''',
    # '''11111111''',
    # '''1951''',
    '''毛泽东是谁''',
    '''什么是冯诺依曼架构''',
    '''What do you think of China?''',
    '''Explaining von Neumann Architecture ''',
    '''我炒股亏钱了心情不好''',
    '''中译英：请问最近的火车站/药店在哪儿？''',
    '''英译中：The last stage went higher and took the Apollo into orbit round the earth.''',
# ]
    '''Why is the Bible the oldest book in history?''',
    '''What vegetables taste good with cabbage?''',
    "将后面的中文翻译为英文：电动汽车比传统汽车更加环保，因为它们不会产生排放，而是由电力驱动，电力是一种可再生能源和清洁的能源。这使得它们成为更可持续的运输选择。",
    "人工智能创业成功的公司有哪些",
    "帮我找一首思恋故乡的诗",
    "不高兴，咋办",
    "有啥冷笑话给我讲一个",
    "给我写一篇短篇小说",
    "我问你个脑筋急转弯：什么花一年四季都开着",
    "I don't like Elon Reeve Musk, how about you",
    "Help me to analyze the development potential of Weibo.",
    "在哪些行业证明人工智能是有效的",
    "中国有多少个省",
    "把后面语句翻译为英文：中国有多少个省，各自有啥特产？",
    "把后面语句翻译为英文：今天的天气真不错啊！你下午有空吗？我想约你一起去吃饭。",
]

for question in questions:
    chat(lm.model,lm.tokenizer,question=question,max_length=512)