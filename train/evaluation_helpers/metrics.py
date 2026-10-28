import torch

#All metrics are computed per query, for a single k, from the ranked list of retrieved tracks:
#   is_relevant:  bool tensor [Q, K_max], True when the j-th retrieved track of the query is a relevant target
#   num_relevant: tensor [Q], the total number of relevant tracks of each query
#and return a tensor [Q] with the score of each query. The mean over the queries is done by aggregate().

def hit_at_k(is_relevant, num_relevant, k):
    #1 if at least one relevant track is among the top k, else 0
    return is_relevant[:, :k].any(dim=1).float()

def recall_at_k(is_relevant, num_relevant, k):
    #fraction of the relevant tracks that are among the top k
    return is_relevant[:, :k].sum(dim=1).float() / num_relevant

def mrr_at_k(is_relevant, num_relevant, k):
    #reciprocal rank of the first relevant track among the top k, 0 if there is none
    relevant = is_relevant[:, :k]
    first_rank = relevant.float().argmax(dim=1) + 1    #argmax returns the first relevant position

    return relevant.any(dim=1).float() / first_rank

def map_at_k(is_relevant, num_relevant, k):
    #average precision at k, normalized by min(k, number of relevant tracks)
    relevant = is_relevant[:, :k].float()
    ranks = torch.arange(1, relevant.shape[1] + 1, device=relevant.device)

    precision_at_i = relevant.cumsum(dim=1) / ranks
    normalization = torch.clamp(num_relevant, max=k).float()

    return (precision_at_i * relevant).sum(dim=1) / normalization

def ndcg_at_k(is_relevant, num_relevant, k):
    #normalized discounted cumulative gain at k, with binary relevance
    relevant = is_relevant[:, :k].float()
    discounts = 1.0 / torch.log2(torch.arange(2, relevant.shape[1] + 2, device=relevant.device).float())

    dcg = (relevant * discounts).sum(dim=1)

    #ideal ranking: all the relevant tracks (at most k) at the top
    ideal_dcg = discounts.cumsum(dim=0)[torch.clamp(num_relevant, max=k) - 1]

    return dcg / ideal_dcg

#all the metrics computed for every run and relevance criterion
METRICS = {
    'hit': hit_at_k,
    'recall': recall_at_k,
    'mrr': mrr_at_k,
    'map': map_at_k,
    'ndcg': ndcg_at_k,
}

def score_run(is_relevant, num_relevant, ks):
    #per query score of every metric at every k, in a dict {'hit@1': tensor [Q], ..., 'ndcg@100': tensor [Q]}
    return {f"{name}@{k}": metric(is_relevant, num_relevant, k) for name, metric in METRICS.items() for k in ks}

def aggregate(per_query_scores):
    #mean over the queries of each score of score_run(), as plain numbers: {'hit@1': 0.12, ...}
    return {column: scores.mean().item() for column, scores in per_query_scores.items()}