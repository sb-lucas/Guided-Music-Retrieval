import os
import sys
import importlib

#sizes of the embeddings of each backbone, read by train.py and evaluate.py. The keys are the names of the folders of the embeddings
BACKBONE_DIMS = {

    'hf_clap': {
        'global_tokens': 512,
        'local_text_tokens': 768,
        'local_audio_tokens': 1024
    },

    'laion_clap': {
        'global_tokens': 512,
        'local_text_tokens': 768,
        'local_audio_tokens': 1024
    },

    'muq': {  #placeholder for future models, currently not used
        'global_tokens': -1,
        'local_text_tokens': -1,
        'local_audio_tokens': -1
    }
}

#module and class of the extractor of each backbone (feature_extraction/extractors). They are imported only when used, so the libraries of the other backbones are not loaded
EXTRACTORS = {
    'hf_clap': ('hf_clap', 'HF_CLAP'),
    'laion_clap': ('laion_clap_extractor', 'LAION_CLAP'),
}

def load_extractor(backbone):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../feature_extraction/extractors')))
    module, class_name = EXTRACTORS[backbone]

    return getattr(importlib.import_module(module), class_name)()
