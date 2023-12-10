import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from vary.model import *
from vary.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from vary.model.plug.transforms import train_transform, test_transform

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = varyQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True)


    model.to(device='cuda',  dtype=torch.bfloat16)


    # TODO download clip-vit in huggingface
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)

    image_processor_high = test_transform

    use_im_start_end = True

    image_token_len = 256

    qs = 'Provide the ocr results of this image.'

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN  + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


    

    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    inputs = tokenizer([prompt])


    image = load_image(args.image_file)
    image_1 = image.copy()
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    image_tensor_1 = image_processor_high(image_1)

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=True,
            num_beams = 1,
            # temperature=0.2,
            streamer=streamer,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria]
            )
        
        # print(output_ids)

        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # # conv.messages[-1][-1] = outputs
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        # outputs = outputs.strip()

        # print(outputs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
