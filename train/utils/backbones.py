#sizes of the embeddings of each backbone, read by train.py and evaluate.py
BACKBONE_DIMS = {

    'CLAP': {
        'global_tokens': 512,
        'local_text_tokens': 768,
        'local_audio_tokens': 1024
    },

    'MUQ': {  #placeholder for future models, currently not used
        'global_tokens': -1,
        'local_text_tokens': -1,
        'local_audio_tokens': -1
    }
}
