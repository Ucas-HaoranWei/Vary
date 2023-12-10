CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


ROOT_PATH = '/data/public/ucaswei/data/'

CONVERSATION_DATA = {

    # pair 4m
    'laion-coco-4m': {
        'images': '',
        'annotations': '',
    }, 

    'cc665k': {
        'images': "/path_to/LLaVA1.5/images/",
        'annotations': "/path_to/LLaVA1.5/llava_v1_5_66k.json",
    },

    'pdf': {
        'images': "",
        'annotations': "",
    },

    'docvqa_train': {
        'images': "",
        'annotations': "",
    },

    'chartqa_train': {
        'images': "",
        'annotations': "",
    },



}