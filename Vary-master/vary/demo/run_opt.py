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

from transformers import TextStreamer


from vary.model.plug.blip_process import BlipImageEvalProcessor

from vary.model.vision_encoder.sam import build_sam_vit_b
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = varyOPTForCausalLM.from_pretrained(model_name)



    model.to(device='cuda',  dtype=torch.bfloat16)

    image_processor_high =  test_transform


    image_token_len = 256


    prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    inputs = tokenizer([prompt])


    image = load_image(args.image_file)
    image_1 = image.copy()

    image_tensor_1 = image_processor_high(image_1).to(torch.bfloat16)


    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = '</s>'
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor_1.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).cuda())],
            do_sample=True,
            num_beams = 1,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
        



        # input_token_len = input_ids.shape[1]
        # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        # outputs = outputs.strip()

        # print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
