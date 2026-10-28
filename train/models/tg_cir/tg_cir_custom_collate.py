import torch
from torch.nn.utils.rnn import pad_sequence

def tg_cir_collate_fn(batch):
    (
        reference_song_local_tokens,
        reference_song_global_tokens,
        modification_text_local_tokens,
        modification_text_global_tokens,
        target_song_local_tokens,
        target_song_global_tokens,
        query_ids,
        target_ids,
    ) = zip(*batch)

    # Pad local tokens (list of [L, D]) → [B, L_max, D]
    reference_song_local_tokens = pad_sequence(reference_song_local_tokens, batch_first=True)
    target_song_local_tokens = pad_sequence(target_song_local_tokens, batch_first=True)
    modification_text_local_tokens = pad_sequence(modification_text_local_tokens, batch_first=True)

    # Global tokens and IDs: fixed-size
    reference_song_global_tokens = torch.stack(reference_song_global_tokens)
    modification_text_global_tokens = torch.stack(modification_text_global_tokens)
    target_song_global_tokens = torch.stack(target_song_global_tokens)
    query_ids = torch.as_tensor(query_ids)
    target_ids = torch.as_tensor(target_ids)

    return (
        reference_song_local_tokens,
        reference_song_global_tokens,
        modification_text_local_tokens,
        modification_text_global_tokens,
        target_song_local_tokens,
        target_song_global_tokens,
        query_ids,
        target_ids,
    )