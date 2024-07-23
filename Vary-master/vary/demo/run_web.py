import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor
from vary.model import varyQwenForCausalLM
from vary.model.plug.transforms import test_transform
from vary.utils.utils import KeywordsStoppingCriteria
from transformers import TextStreamer
import datetime
import uuid
from vary.utils.utils import disable_torch_init
from vary.utils.conversation import conv_templates, SeparatorStyle
import os
import argparse

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

parser = argparse.ArgumentParser()
parser.add_argument('--host', nargs='?', const='0.0.0.0', help='Host for the server')
parser.add_argument('--model-name', default='./vary/model/vary-llava80k/', help='Model name for processing')
args = parser.parse_args()

disable_torch_init()

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

model = varyQwenForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True)


model.to(device='cuda',  dtype=torch.bfloat16)


# TODO download clip-vit in huggingface
image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)

def eval_model(image_file):

    image_processor_high = test_transform

    use_im_start_end = True

    image_token_len = 256

    qs = 'Provide the ocr results of this image.'

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN  + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    inputs = tokenizer([prompt])


    image = Image.open(image_file).convert('RGB')
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

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # conv.messages[-1][-1] = outputs
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print(outputs)
        return outputs

def gradio_interface(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f"{timestamp}_{uuid.uuid4()}.jpg"

    temp_image_path = os.path.join("/tmp", unique_filename)
    image.save(temp_image_path)

    model_name = args.model_name
    result = eval_model(temp_image_path)
    os.remove(temp_image_path)

    return result

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_button = gr.Button("Submit")
        with gr.Column():
            output_text = gr.Textbox(label="Model Output")

    submit_button.click(gradio_interface, inputs=image_input, outputs=output_text)

if args.host is not None:
    demo.launch(server_name=args.host)
else:
    demo.launch()