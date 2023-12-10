#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from vary.utils.constants import *

from vary.model.plug.blip_process import BlipImageEvalProcessor

from vary.model.llm.qwen.modeling_qwen import QWenLMHeadModel, QWenModel

from vary.model.llm.qwen.configuration_qwen import QWenConfig
from vary.model.vision_encoder.sam import build_sam_vit_b
from vary.model.plug.transforms import train_transform, test_transform


class varyConfig(QWenConfig):
    model_type = "vary"


class varyQwenModel(QWenModel):
    config_class = varyConfig

    def __init__(self, config: QWenConfig):
        super(varyQwenModel, self).__init__(config)
        # TODO download the clip-vit in huggingface
        self.vision_tower = CLIPVisionModel.from_pretrained('/cache/vit-large-patch14/')

        self.vision_tower_high = build_sam_vit_b()    #  build_sam_vit_b(checkpoint = 'xxxx') for train

        self.mm_projector =  nn.Linear(1024, 2048)
        self.mm_projector_vary =  nn.Linear(1024, 2048)

    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=False,
        vision_select_layer=-1,
        dtype=torch.float16,
        device="cuda"
    ):

        # 224*224
        # TODO download the clip-vit in huggingface
        image_processor = CLIPImageProcessor.from_pretrained('/cache/vit-large-patch14/')
        # 1024*1024
        image_processor_high = train_transform

        self.vision_tower = self.vision_tower.to(dtype=dtype, device=device)

        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)

        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)
        self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype, device=device)

        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len

        self.config.use_im_start_end = True

        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor=image_processor,
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,

        )

    def embed_tokens(self, x):
        return self.wte(x)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        # orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight[:-self.num_new_tokens] = orig_embeds_params[:-self.num_new_tokens].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # inputs_embeds = self.wte(input_ids)


        vision_tower = getattr(self, 'vision_tower', None)
        vision_tower_high = getattr(self, 'vision_tower_high', None)


        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:

            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            # im_patch_token = getattr(self.config, "im_patch_token", -1)
            # im_start_token = getattr(self.config, "im_start_token", -1)
            # im_end_token = getattr(self.config, "im_end_token", -1)
            # freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)

            im_patch_token = 151859

            im_start_token = 151857

            im_end_token = 151858


            image_features_1 = []
            image_features_2 = []
            for image in images:

                with torch.set_grad_enabled(False):
                    image_forward_out = vision_tower(image[0], output_hidden_states=True)
                    select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
                    image_feature = select_hidden_state[:, 1:]  # 256*1024
                with torch.set_grad_enabled(False):
                    cnn_feature = vision_tower_high(image[1])
                    cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # 256*1024

                image_features_1.append(image_feature)
                image_features_2.append(cnn_feature)


            if type(images) is list:
                image_features_1 = [self.mm_projector(image_feature) for image_feature in image_features_1]
                image_features_2 = [self.mm_projector_vary(image_feature) for image_feature in image_features_2]
                image_features = [torch.cat((image_feature[0], image_feature[1]), dim=-1) for image_feature in zip(image_features_1, image_features_2)]
            else:

                raise NotImplementedError


            # dummy_image_features = torch.zeros(256, 4096, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features_1 = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features_2 = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features_1 = self.mm_projector(dummy_image_features_1)
            dummy_image_features_2 = self.mm_projector_vary(dummy_image_features_2)
            dummy_image_features = torch.cat((dummy_image_features_1, dummy_image_features_2), dim=-1)
            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        # if orig_embeds_params is not None:
                        #     cur_new_input_embeds = torch.cat(
                        #         (
                        #             cur_input_embeds[:image_start_token_pos].detach(), 
                        #             cur_input_embeds[image_start_token_pos:image_start_token_pos+1], 
                        #             per_cur_image_features, 
                        #             cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], 
                        #             cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()
                        #         ), 
                        #         dim=0
                        #     )
                        # else:
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ), 
                            dim=0
                        )


                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(varyQwenModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class varyQwenForCausalLM(QWenLMHeadModel):
    config_class = varyConfig
    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(QWenLMHeadModel, self).__init__(config)
        self.transformer = varyQwenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, varyQwenModel):
    #         module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)


        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict
            
        )



        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # logits

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # print(loss)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self, 
        tokenizer, 
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        config = self.get_model().config


        self.resize_token_embeddings(len(tokenizer))


        config.im_patch_token = 151859

        config.use_im_start_end = True

        # add image start token <im_start> and end token <im_end>
        if config.use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            # config.im_start_token, config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            config.im_start_token, config.im_end_token = 151857, 151858


AutoConfig.register("vary", varyConfig)
AutoModelForCausalLM.register(varyConfig, varyQwenForCausalLM)
