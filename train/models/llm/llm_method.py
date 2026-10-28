import os
import sys
import torch
import torch.nn.functional as F

from models.base_method import BaseMethod
from models.llm.llm import LLM_Based

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../feature_extraction')))
from extractors.clap import CLAP

# TODO: compose generates one description at a time, with sampling (temperature 1.5), so it is slow and not reproducible
#       unless torch is seeded before it. Batched generation would be faster.
# TODO: the tags of every reference song must be in the classifications file, including the generated MTSQ audios.
# TODO: llm_utils.py (the old LLM_fusion) is dead code now that this class replaces it.
class LLMMethod(BaseMethod):
    #zero-shot: the LLM writes a description of the target from the tags of the reference song and the modification text,
    #and CLAP encodes that description. The pool is encoded by CLAP directly, like the other global methods (BaseMethod.encode_pool)
    def __init__(self, loss_tau, global_dim, classifications_path, hf_cache_dir, model_name="Qwen/Qwen3-8B"):
        #set before super().__init__(), which calls build_model()
        self.classifications_path = classifications_path
        self.hf_cache_dir = hf_cache_dir
        self.model_name = model_name

        super().__init__(loss_tau, global_dim)
        self.clap = CLAP()

    def build_model(self):
        return LLM_Based(self.model_name, self.classifications_path, self.hf_cache_dir)

    def to(self, device):
        #the LLM places itself on the GPUs (device_map="auto") and CLAP chooses its own device, so only the device of the output is kept
        self.device = torch.device(device)

    def eval(self):
        #from_pretrained already returns the models in eval mode
        pass

    def training_step(self, *batch):
        raise NotImplementedError("LLMMethod is zero-shot and is not trained")

    def load_checkpoint(self, path, checkpoint_name='checkpoint'):
        raise NotImplementedError("LLMMethod is zero-shot and has no checkpoint")

    @torch.no_grad()
    def compose(self, queries):
        #queries: dict from EvaluationSet.get_queries. This method reads the strings, not the embeddings
        composed = []
        for song, modification_text in zip(queries['reference_songs'], queries['modification_texts']):
            target_description = self.model.get_target_description(song, modification_text)

            global_text, _ = self.clap.get_text_tokens(target_description)
            composed.append(F.normalize(global_text.detach().float().squeeze(0).cpu(), dim=-1))

        return torch.stack(composed).to(self.device)
