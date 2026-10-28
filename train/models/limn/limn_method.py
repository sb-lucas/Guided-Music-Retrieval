import torch
from torch.nn.utils.rnn import pad_sequence

from models.base_method import BaseMethod
from models.limn.limn import LIMN
from models.limn.limn_losses import LIMN_Loss
from models.limn.limn_custom_collate import limn_collate_fn

# TODO: compose and encode_pool follow the interface of BaseMethod (the queries dict, the pool tensor), but training_step
#       still takes the tuple of limn_collate_fn, with local tokens, lengths and ids.
# TODO: the padding of local tokens with different lengths (modification text, and audio if it varies) still has to be
#       handled. LIMN's transformer assumes the local tokens of a sample come packed before the factor tokens, but a batch
#       pads them to the longest one, so batched results differ from one-by-one results when the lengths differ.
# TODO: latent_factor_ranking_loss (limn_losses.py) is a second batch-wide softmax with diagonal positives and does not
#       receive query_id and target_id yet, so false negatives are only masked in the BBCL term.
# TODO: LIMN.__init__ takes num_layers but never uses it: TransformerBlock is built with its default of 2 layers.
class LIMNMethod(BaseMethod):
    def __init__(self, loss_tau, global_dim, audio_local_dim, text_local_dim):
        super().__init__(loss_tau, global_dim, audio_local_dim, text_local_dim)
        self.needs_local_tokens = True
        self.loss_func = LIMN_Loss(loss_tau)
        self.custom_collate_fn = limn_collate_fn

    def build_model(self):
        return LIMN(embed_dim=self.global_dim, audio_token_dim=self.audio_local_dim, text_token_dim=self.text_local_dim)

    # the batch is what limn_collate_fn returns
    def training_step(self, ref_local, ref_global, mod_text_local, mod_text_global, target_local, target_global, ref_len, mod_text_len, target_len, query_id, target_id):
        ref_local, ref_global, mod_text_local, mod_text_global, target_local, target_global, ref_len, mod_text_len, target_len, query_id, target_id = self.to_device(
            ref_local, ref_global, mod_text_local, mod_text_global, target_local, target_global, ref_len, mod_text_len, target_len, query_id, target_id)

        query_len = ref_len + mod_text_len
        x_hat_q, a_hat_q, x_hat_t, a_hat_t = self.model(ref_global, mod_text_global, target_global, ref_local, mod_text_local, target_local, query_len, target_len)

        loss = self.loss_func(a_hat_q, a_hat_t, self.model.condition_masks_q, self.model.condition_masks_t, x_hat_q, x_hat_t, query_id, target_id)

        return loss

    @torch.no_grad()
    def compose(self, queries):
        #queries: dict from EvaluationSet.get_queries, built with local tokens
        self.eval()     #dropout must be off to compose queries
        ref_global, mod_text_global, ref_local, mod_text_local, ref_len, mod_text_len = self.to_device(
            queries['ref_global'], queries['mod_text_global'], queries['ref_local'], queries['mod_text_local'], queries['ref_len'], queries['mod_text_len'])

        batch_size = ref_global.size(0)

        #the target half of the model runs on dummy inputs, only the query half is used
        dummy_global = torch.zeros_like(ref_global)
        dummy_local = torch.zeros(batch_size, 1, ref_local.size(-1), device=self.device)
        dummy_len = torch.ones(batch_size, dtype=torch.long, device=self.device)

        _, a_hat_q, _, _ = self.model(ref_global, mod_text_global, dummy_global, ref_local, mod_text_local, dummy_local, ref_len + mod_text_len, dummy_len)

        return a_hat_q

    @torch.no_grad()
    def encode_pool(self, audio_cache, song_ids, batch_size=256):
        self.eval()     #dropout must be off to encode the pool

        embeddings_matrix = []
        for start in range(0, len(song_ids), batch_size):
            chunk = song_ids[start:start + batch_size]
            n = len(chunk)

            target_global = torch.stack([audio_cache[song_id][0] for song_id in chunk])
            target_local = [audio_cache[song_id][1] for song_id in chunk]
            target_len = torch.tensor([tokens.size(0) for tokens in target_local])
            target_local = pad_sequence(target_local, batch_first=True)
            target_global, target_local, target_len = self.to_device(target_global, target_local, target_len)

            #the query half of the model runs on dummy inputs, only the target half is used
            dummy_global = torch.zeros_like(target_global)
            dummy_ref_local = torch.zeros(n, 1, target_local.size(-1), device=self.device)
            dummy_text_local = torch.zeros(n, 1, self.text_local_dim, device=self.device)
            dummy_query_len = torch.full((n,), 2, dtype=torch.long, device=self.device)

            _, _, _, a_hat_t = self.model(dummy_global, dummy_global, target_global, dummy_ref_local, dummy_text_local, target_local, dummy_query_len, target_len)

            embeddings_matrix.append(a_hat_t.cpu())

        return torch.cat(embeddings_matrix)
