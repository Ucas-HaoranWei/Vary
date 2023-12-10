
import io
import os
import copy
import json
import logging
import torch
import random

from typing import List, Optional, Tuple, Union, Dict, Sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from vary.data.base_dataset import BaseDataset
from vary.utils.constants import *
from vary.utils import conversation as conversation_lib


class CaptionDataset(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, datasets, tokenizer, multimodal_cfg):
        super(CaptionDataset, self).__init__(datasets, tokenizer, multimodal_cfg)
        # v0 version format conversation
        conversation_lib.default_conversation = conversation_lib.conv_templates["default"]
        logging.warning("Formatting inputs into conversation type: v0-fixed")
        logging.warning("Loading data...")

        list_data_dict = []
        list_image_path = []
        for name in datasets.split("+"):
            dataset = CONVERSATION_DATA[name] # in vary.utils

            data_path = dataset['annotations']
            data = json.load(open(data_path, "r"))

            list_data_dict.extend(data)

            image_path = dataset['images']
            list_image_path.extend([image_path] * len(data))

            logging.warning(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        logging.warning(f"{len(list_data_dict)} conversations in total.")
        a_new_list = list(zip(list_data_dict, list_image_path))
        random.shuffle(a_new_list)
        list_data_dict_new, list_image_path_new = zip(*a_new_list)
        self.list_data_dict = list_data_dict_new
        self.list_image_path = list_image_path_new
        self.im_patch_token, self.im_start_token, self.im_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    
    def multimodal_processor(self, sources):
        for source in sources:

            source[0]['value'] = DEFAULT_IMAGE_TOKEN
            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
                if self.multimodal_cfg['use_im_start_end']:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources

    def _tokenize_fn(self, strings):
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]

        
        for idx, ii in enumerate(input_ids):
            if ii[-1] != 2:
                input_ids[idx][-1] = 2
                labels[idx][-1] = 2

        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def _mask_targets(self, target, tokenized_lens, speakers):
        # cur_idx = 0
        cur_idx = tokenized_lens[0]
        tokenized_lens = tokenized_lens[1:]
        target[:cur_idx] = IGNORE_INDEX
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker.lower() == "human":
                target[cur_idx:tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len


    def _add_speaker_and_signal(self, header, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "</s>"
        END_SIGNAL = "\n"
        conversation = header
        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() == "human":
                from_str = conversation_lib.default_conversation.roles[0]
            else:
                from_str = conversation_lib.default_conversation.roles[1]

            sentence["value"] = sentence["value"] + END_SIGNAL
            if get_conversation:
                conversation += sentence["value"]
        conversation += BEGIN_SIGNAL
        return conversation

    def token_processor(self, sources):
        """
        Given a list of sources, each is a conversation list. This transform:
        1. Add signal '### ' at the beginning each sentence, with end signal '\n';
        2. Concatenate conversations together;
        3. Tokenize the concatenated conversation;
        4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
        """
        # add end signal and concatenate together
        conversations = []
        for source in sources:
            header = ''
            conversation = self._add_speaker_and_signal(header, source)
            conversations.append(conversation)

        conversations_tokenized = self._tokenize_fn(conversations)
        input_ids = conversations_tokenized["input_ids"]
        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            tokenized_lens = self._tokenize_fn([header] + [s["value"] for s in source])["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            self._mask_targets(target, tokenized_lens, speakers)

        return dict(input_ids=input_ids, labels=targets)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # data = self.list_data_dict[i]
        data = copy.deepcopy(self.list_data_dict[i])

        if isinstance(data, dict):
            if 'image' in data:
                image_path = self.list_image_path[i]
                image_file = data['image']
                # TODO this is a bug, because some json has wrong path

                try:
                    image = Image.open(image_path + image_file).convert('RGB')
                except:
                    print(f'cannot identify image file {image_path+image_file}.')
                    return self.__getitem__(0)

                try:
                    image, image_high = self.image_processor(image)
                except:
                    print(f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
                    return self.__getitem__(0)

            conversations = self.multimodal_processor([data["conversations"]])
        else:
            conversations = [data]

        # align with fastchat & llava here, put the conversation into a list for tokenization
        data_dict = self.token_processor(conversations)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        if isinstance(data, dict) and 'image' in data:
            data_dict['image'] = [image]
            data_dict['image_high'] = [image_high]
        else:
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            # TODO sam is 1024*1024
            data_dict['image_high'] = [torch.zeros(3, 1024, 1024)]
        return data_dict

