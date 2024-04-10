
CUDA_VISIBLE_DEVICES=0 python main.py --model /data01/ssd/llama2-7b-hf/  --epochs 20 --output_dir ./log/llama--7b-w4a16-global_debug --eval_ppl --wbits 4 --abits 4 --lwc --let --act-scales ./act_scales/llama2-7b.pt --act-shifts ./act_shifts/llama2-7b.pt --net Llama-2-7b

CUDA_VISIBLE_DEVICES=0 python main.py --model /data01/ssd/llama2-7b-hf/  --epochs 20 --output_dir ./log/llama--7b-w4a16-global_debug --eval_ppl --wbits 4 --abits 16 --lwc --let --act-scales ./act_scales/llama2-7b.pt --act-shifts ./act_shifts/llama2-7b.pt --net Llama-2-7b