
<h3><a href="">Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models</a></h3>

[Haoran Wei*](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), [Lingyu Kong*](), [Jinyue Chen](), [Liang Zhao](), [Zheng Ge](https://joker316701882.github.io/), [Jinrong Yang](https://yancie-yjr.github.io/), [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), [Chunrui Han](), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)
	
<a href="https://varybase.github.io/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href="#"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 
<a href="http://region-31.seetacloud.com:22701/"><img src="https://img.shields.io/badge/demo-blue"></a> 

<p align="center">
<img src="assets/logo.jpg" style="width: 200px" align=center>
</p>

## Release
- [2023/12/11] The paper will be released in the next two days.
- [2023/12/11] We released the online demo, have fun! 
- [2023/12/11] We released the codes of Vary (train and inference)! 

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to use that follow the license agreement of LLaMA, Vicuna, GPT-4, Qwen, and LLaVA. 


## Contents
- [Install](#install)
- [Vary Weights](#vary-weights)
- [Demo](#Demo)
- [Train](#train)

## Install
1. Clone this repository and navigate to Vary folder
```bash
git clone https://github.com/Ucas-HaoranWei/Vary.git
cd Vary
```
2. Install Package
```Shell
conda create -n vary python=3.10 -y
conda activate vary
pip install e .
```

3. Install Flash-Attention
```
pip install ninja
pip install flash-attn --no-build-isolation
```

## Vary Weights
- We released the Vary-base (qwen) after [llava-80k](https://github.com/haotian-liu/LLaVA) SFT, you can download in [Baiduyun](https://pan.baidu.com/s/1HrJEXkZsBWNp3dTvnv2OSA), code: vary, we will upload the weights to other Netdisk soon!
- Download the CLIP-VIT-L in [huggingface](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
  
## Demo
1.Update the CLIP-VIT path in the codes (/cache/vit-large-patch14/) to your path.
2.
```Shell
python vary/demo/run_qwen_vary.py  --model-name  /vary/model/path/ --image-file /an/image/file.png
```
## Train
- We currently do not plan to open source the weights of the intermediate.
- However, we release the train codes. So you can train on your own dataset.
If you want to do this, you can try this:
1. For Vary-base (one machine, if you have multiple machines you need prepare your hostfile)
```Shell
deepspeed   Vary/train/train_qwen_vary.py  --deepspeed /Vary/zero_config/zero2.json
            --model_name_or_path /Qwen-7B/path/
            --vision_tower /vit-large-patch14/path/
            --freeze_vision_tower True
            --freeze_lm_model False
            --vision_select_layer  -2
            --use_im_start_end True
            --bf16 True
            --per_device_eval_batch_size 4
            --gradient_accumulation_steps 1
            --evaluation_strategy "no"
            --save_strategy "steps"
            --save_steps 5000
            --save_total_limit 1
            --weight_decay 0.
            --warmup_ratio 0.03
            --lr_scheduler_type "cosine"
            --logging_steps 1 --tf32 True
            --model_max_length 4096
            --gradient_checkpointing True
            --dataloader_num_workers 4
            --report_to none
            --per_device_train_batch_size 4
            --num_train_epochs 1
            --learning_rate 5e-5
            --datasets  data_name1+data_name2+data_name3
            --output_dir /path/to/output/
```
2. For Vary-tiny
```Shell
deepspeed   Vary/train/train_opt.py  --deepspeed /Vary/zero_config/zero2.json
            --model_name_or_path /opt125m/path/
            --conversation_version opt
            --freeze_vision_tower False
            --freeze_lm_model False
            --use_im_start_end True
            --bf16 True
            --per_device_eval_batch_size 4
            --gradient_accumulation_steps 1
            --evaluation_strategy "no"
            --save_strategy "steps"
            --save_steps 5000
            --save_total_limit 1
            --weight_decay 0.
            --warmup_ratio 0.03
            --lr_scheduler_type "cosine"
            --logging_steps 1 --tf32 True
            --model_max_length 4096
            --gradient_checkpointing True
            --dataloader_num_workers 4
            --report_to none
            --per_device_train_batch_size 16
            --num_train_epochs 1
            --learning_rate 5e-5
            --datasets  data_name1+data_name2+data_name3
            --output_dir /path/to/output/
```


## Contact
If you have any questions related to the code or the paper, feel free to email Haoran Wei (`weihaoran18@mails.ucas.ac.cn`).

## Acknowledgement
- [LLaVA](https://github.com/lm-sys/FastChat): the codebase we built upon!
- [Qwen](https://github.com/QwenLM/Qwen): the LLM base model of Vary, which is good at both English and Chinese!


## Citation
If you find our work useful in your research, please consider citing Vary:
```tex
placeholder for paper
```
