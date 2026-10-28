import torch
import torch.nn.functional as F

class Retriever:
    #given a method and an evaluation set, retrieves from the pool of the set the tracks closest to the composed query of each triplet

    def __init__(self, method, eval_set, top_k=100, batch_size=1024):
        self.method = method
        self.eval_set = eval_set
        self.top_k = top_k
        self.batch_size = batch_size

        #the pool is encoded once, in the order of the positions of the evaluation set.
        #exact retrieval by brute force on the GPU: with unit-norm embeddings, the cosine similarity is the dot product
        pool = self.method.encode_pool(self.eval_set.audio_cache, self.eval_set.song_ids)
        self.pool_embeddings = F.normalize(pool.float(), dim=-1).to(self.method.device)

    @torch.no_grad()
    def retrieve_top_k(self, composed, reference_positions):
        #positions in the pool of the top_k most similar tracks of each query, without its own reference track. shape [B, top_k]
        composed = F.normalize(composed.float().to(self.method.device), dim=-1)

        scores = composed @ self.pool_embeddings.T   #[B, number of tracks in the pool]
        scores[torch.arange(len(composed), device=scores.device), reference_positions.to(scores.device)] = float('-inf')

        return scores.topk(min(self.top_k, scores.size(1)), dim=1).indices

    @torch.no_grad()
    def retrieve(self):
        #positions in the pool of the top_k tracks retrieved for each triplet of the evaluation set, best first. shape [number of triplets, top_k]
        n_triplets = len(self.eval_set)
        top_k_positions = torch.zeros(n_triplets, min(self.top_k, self.pool_embeddings.size(0)), dtype=torch.long)

        for start in range(0, n_triplets, self.batch_size):
            end = min(start + self.batch_size, n_triplets)
            print(f"{start}/{n_triplets}", end='\r')

            queries = self.eval_set.get_queries(start, end)
            composed = self.method.compose(queries)

            top_k_positions[start:end] = self.retrieve_top_k(composed, self.eval_set.reference_positions[start:end]).cpu()

        return top_k_positions
