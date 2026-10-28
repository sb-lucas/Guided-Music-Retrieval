#A relevance criterion is a function of the evaluation set, not of the run: it gives the tracks that count as correct
#for each triplet, so it is computed once and used for every method.
#   criterion(eval_set) -> relevant_positions [Q, R_max], positions in the pool of the relevant tracks of each triplet, padded with -1
#The reference track of a triplet is never retrieved (the retriever masks it), so it must not be among its relevant tracks.

def exact_target(eval_set):
    #only the target track of the triplet is relevant
    return eval_set.target_positions[:, None]

#the criteria that evaluate.py can run, by the name used in the results file. New criteria (same source, tags that satisfy
#the modification, ...) are added here
CRITERIA = {
    'exact': exact_target,
}

def build_relevance(top_k_positions, relevant_positions):
    """
    top_k_positions: [Q, K_max] positions in the pool of the retrieved tracks, best first
    relevant_positions: [Q, R_max] positions in the pool of the relevant tracks of each query, unique, padded with -1
    Returns is_relevant [Q, K_max] and num_relevant [Q].
    """
    #the comparison is a tensor [Q, K_max, R_max], so a criterion with many relevant tracks per query needs chunking over Q
    is_relevant = (top_k_positions[:, :, None] == relevant_positions[:, None, :]).any(dim=-1)
    num_relevant = (relevant_positions >= 0).sum(dim=1)

    return is_relevant, num_relevant
