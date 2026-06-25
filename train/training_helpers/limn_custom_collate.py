import torch
from torch.nn.utils.rnn import pad_sequence

def limn_collate_fn(batch):
    (
        reference_song_local_tokens,
        reference_song_global_tokens,
        modification_text_local_tokens,
        modification_text_global_tokens,
        target_song_local_tokens,
        target_song_global_tokens,
    ) = zip(*batch)

    # Compute original lengths before padding
    reference_lengths = torch.tensor([t.size(0) for t in reference_song_local_tokens])
    modification_lengths = torch.tensor([t.size(0) for t in modification_text_local_tokens])
    target_lengths = torch.tensor([t.size(0) for t in target_song_local_tokens])

    # Pad local tokens (list of [L, D]) → [B, L_max, D]
    reference_song_local_tokens = pad_sequence(reference_song_local_tokens, batch_first=True)
    target_song_local_tokens = pad_sequence(target_song_local_tokens, batch_first=True)
    modification_text_local_tokens = pad_sequence(modification_text_local_tokens, batch_first=True)

    # Global tokens and IDs: fixed-size
    reference_song_global_tokens = torch.stack(reference_song_global_tokens)
    modification_text_global_tokens = torch.stack(modification_text_global_tokens)
    target_song_global_tokens = torch.stack(target_song_global_tokens)

    return (
        reference_song_local_tokens,
        reference_song_global_tokens,
        modification_text_local_tokens,
        modification_text_global_tokens,
        target_song_local_tokens,
        target_song_global_tokens,
        reference_lengths,
        modification_lengths,
        target_lengths,
    )