import torch

from models.base_method import BaseMethod
from models.tg_cir.tg_cir import TG_CIR
from models.tg_cir.tg_cir_losses import TG_CIR_Loss
from models.tg_cir.tg_cir_custom_collate import tg_cir_collate_fn

# TODO: compose and encode_pool follow the interface of BaseMethod (the queries dict, the pool tensor), but training_step
#       still takes the tuple of tg_cir_collate_fn, with local tokens and ids.
# TODO: the padding of local tokens with different lengths (modification text, and audio if it varies) still has to be
#       handled. TG-CIR's local extractor receives no lengths, so the padded zero tokens take part in its pooling
#       and the result depends on which texts share a batch.
# TODO: TG-CIR uses one local extractor for both audio and text tokens, so it cannot run when their sizes differ
#       (1024-d audio and 768-d text with larger_clap_music). build_model raises a ValueError. Fixing it is a model change.
# TODO: query_id and target_id reach training_step but not the loss. Still to decide where TG_CIR_Loss uses them:
#       StudentBBCL, TeacherBBCL and the KL term (TargetSimilarityGuidedLoss).
# TODO: loss_tau is ignored. The sub-losses have a fixed tau of 1, and TG_CIR_Loss(tau=0.5) ignores its own argument.
# TODO: tg_cir_utils.py (the old inference and evaluation functions) is dead code now that this class replaces it.
class TG_CIRMethod(BaseMethod):
    def __init__(self, loss_tau, global_dim, audio_local_dim, text_local_dim):
        super().__init__(loss_tau, global_dim, audio_local_dim, text_local_dim)    #loss_tau is not used here: TG_CIR_Loss has its own temperatures
        self.needs_local_tokens = True
        self.loss_func = TG_CIR_Loss()
        self.custom_collate_fn = tg_cir_collate_fn

    def build_model(self):
        #one local extractor is shared by the audio and the text tokens, so both must have the same size
        if self.audio_local_dim != self.text_local_dim:
            raise ValueError(f"TG-CIR shares one local extractor between audio and text tokens, but their sizes differ: {self.audio_local_dim} and {self.text_local_dim}")

        return TG_CIR(global_dim=self.global_dim, local_in_dim=self.audio_local_dim)

    # the batch is what tg_cir_collate_fn returns
    def training_step(self, ref_local, ref_global, mod_text_local, mod_text_global, target_local, target_global, query_id, target_id):
        #query_id and target_id are not used yet: TG_CIR_Loss does not take them
        ref_local, ref_global, mod_text_local, mod_text_global, target_local, target_global = self.to_device(
            ref_local, ref_global, mod_text_local, mod_text_global, target_local, target_global)

        E_r, E_m, E_t, E_c_hat, m_k_hat, m_r_hat, E_c_tilde, m_k_tilde, m_r_tilde = self.model(ref_global, mod_text_global, target_global, ref_local, mod_text_local, target_local)

        loss = self.loss_func(E_r, E_m, E_t, E_c_hat, m_k_hat, m_r_hat, E_c_tilde, m_k_tilde, m_r_tilde)

        return loss

    @torch.no_grad()
    def compose(self, queries):
        #queries: dict from EvaluationSet.get_queries, built with local tokens
        self.eval()     #dropout must be off to compose queries
        ref_global, mod_text_global, ref_local, mod_text_local = self.to_device(
            queries['ref_global'], queries['mod_text_global'], queries['ref_local'], queries['mod_text_local'])

        #without the target, the model returns only the composed attributes of the student branch
        E_c_hat = self.model(ref_global, mod_text_global, None, ref_local, mod_text_local, None)

        return torch.mean(E_c_hat, dim=1)

    @torch.no_grad()
    def encode_pool(self, audio_cache, song_ids, batch_size=256):
        self.eval()     #dropout must be off to encode the pool

        embeddings_matrix = []
        for start in range(0, len(song_ids), batch_size):
            chunk = song_ids[start:start + batch_size]

            target_global = torch.stack([audio_cache[song_id][0] for song_id in chunk])
            target_local = torch.stack([audio_cache[song_id][1] for song_id in chunk])
            target_global, target_local = self.to_device(target_global, target_local)

            #without the reference, the model returns only the attributes of the target
            E_t = self.model(None, None, target_global, None, None, target_local)

            embeddings_matrix.append(torch.mean(E_t, dim=1).cpu())

        return torch.cat(embeddings_matrix)
