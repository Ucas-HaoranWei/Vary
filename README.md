<h3><a href="">Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models</a></h3>
<a href="https://varybase.github.io/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href="https://arxiv.org/abs/2312.06109"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 
<a href="http://region-31.seetacloud.com:22701/"><img src="https://img.shields.io/badge/demo-blue"></a> 
<a href="https://zhuanlan.zhihu.com/p/671420712"><img src="https://img.shields.io/badge/zhihu-yellow"></a> 

[Haoran Wei*](https://scholar.google.com/citations?user=J4naK0MAAAAJ&hl=en), Lingyu Kong*, Jinyue Chen, Liang Zhao, [Zheng Ge](https://joker316701882.github.io/), [Jinrong Yang](https://yancie-yjr.github.io/), [Jianjian Sun](https://scholar.google.com/citations?user=MVZrGkYAAAAJ&hl=en), Chunrui Han, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&hl=en)
	


<p align="center">
<img src="assets/logo.jpg" style="width: 200px" align=center>
</p>

## Release
- [2024/4/21] 🔥🔥🔥 For OneChart, we have released the web demo in [Project Page](https://onechartt.github.io/). Have fun!!
- [2024/4/21] 🔥🔥🔥 We present a Vary-tiny LAVIS codebase (for training from scratch) and the Vary-600k dataset (300K English and 300K Chinese pages) [here](https://github.com/Ucas-HaoranWei/Vary-tiny-600k) !!!
- [2024/4/15]🔥🔥🔥We release a chart parsing model OneChart [here](https://github.com/LingyvKong/OneChart).
- [2024/4/12]🔥🔥🔥We will release a chart parsing model based on Vary-tiny next week. The model supports both English and Chinese charts.
- [2024/3/16]🔥🔥🔥I found many friends very interested in Vary-tiny(OPT-125M), so I opened source it [here](https://huggingface.co/HaoranWei/Vary-tiny-opt125M/tree/main), a PDF-dense OCR and object detection version.
- [2023/1/23]🔥🔥🔥We release the Vary-toy [here](https://github.com/Ucas-HaoranWei/Vary-toy). Besides, we show the super good Vary-family results [here](https://github.com/Ucas-HaoranWei/Vary-family).
- [2023/12/29]🔥🔥🔥We will release a new model (a small-size Vary, about 2B) at the beginning of next month and introduce a new feature (object detection). Our online demo will be temporarily closed to prepare for the deployment of the new model.
- [2023/12/11] We released the online demo, have fun! 
- [2023/12/11] We released the codes of Vary (train and inference)! 

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only. They are also restricted to use that follow the license agreement of LLaMA, Vicuna, GPT-4, Qwen, and LLaVA. 


## Contents
- [Install](#install)
- [Vary Weights](#vary-weights)
- [Demo](#Demo)
- [Train](#train)

## Install
1. Clone this repository and navigate to the Vary folder
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
- If you are in urgent need of weights for your research recently, please contact me by email. 
- Download the CLIP-VIT-L in [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14/tree/main)
  
## Demo
1. Update the CLIP-VIT path in the codes (/cache/vit-large-patch14/) to your path.

2.
```Shell
python vary/demo/run_qwen_vary.py  --model-name  /vary/model/path/ --image-file /an/image/file.png
```
## Train
- We currently do not plan to open source the weights of the intermediate.
- However, we release the train codes. So you can train on your own dataset.
If you want to do this, you can try this:
1. For Vary-base (one machine, if you have multiple machines you need to prepare your host file)
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
If you have any questions related to the code or the paper, feel free to email (`weihaoran18@mails.ucas.ac.cn`).

## Acknowledgement
- [LLaVA](https://github.com/lm-sys/FastChat): the codebase we built upon!
- [Qwen](https://github.com/QwenLM/Qwen): the LLM base model of Vary, which is good at both English and Chinese!




## Citation
If you find our work useful in your research, please consider citing Vary:
```bibtex
@article{wei2023vary,
  title={Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models},
  author={Wei, Haoran and Kong, Lingyu and Chen, Jinyue and Zhao, Liang and Ge, Zheng and Yang, Jinrong and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2312.06109},
  year={2023}
}

@article{wei2024small,
  title={Small Language Model Meets with Reinforced Vision Vocabulary},
  author={Wei, Haoran and Kong, Lingyu and Chen, Jinyue and Zhao, Liang and Ge, Zheng and Yu, En and Sun, Jianjian and Han, Chunrui and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2401.12503},
  year={2024}
}
```
