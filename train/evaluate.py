import os
import json
import torch

from models.concatenation.concatenation_method import ConcatenationMethod
from models.combiner.combiner_method import CombinerMethod
from models.limn.limn_method import LIMNMethod
from models.tg_cir.tg_cir_method import TG_CIRMethod
from models.baseline.baseline_method import BaselineMethod
from models.llm.llm_method import LLMMethod

from evaluation_helpers.evaluation_set import EvaluationSet
from evaluation_helpers.retriever import Retriever
from evaluation_helpers.relevance import CRITERIA, build_relevance
from evaluation_helpers.metrics import score_run, aggregate
from evaluation_helpers.save_results_csv import save_results_csv
from evaluation_helpers.arg_parser import parse_args
from utils.backbones import BACKBONE_DIMS

from dotenv import load_dotenv

load_dotenv()

RUNS_FOLDER = os.getenv("RUNS_FOLDER")
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")     #not in the .env yet, so the results go to ./results

#values of k of the metrics
KS = [1, 2, 5, 10, 20, 50, 100]

#the Method class of each model name. The first four are trained (runs/<run>/settings.json has the name), the last two are zero-shot
methods = {
    'concatenation': ConcatenationMethod,
    'combiner': CombinerMethod,
    'baseline': BaselineMethod,
    #'limn': LIMNMethod,
    #'tg_cir': TG_CIRMethod,
    #'llm': LLMMethod,
}

#zero-shot methods by the name used in the results: (name in methods, arguments of the constructor)
zero_shot = {
    'audio_only': ('baseline', dict(alpha=1.0, beta=0.0)),
    'text_only': ('baseline', dict(alpha=0.0, beta=1.0)),
    'sum': ('baseline', dict(alpha=0.5, beta=0.5)),
    'llm': ('llm', dict()),
}

#columns with the information of each model, before its scores. Zero-shot methods leave the training ones empty
INFO_COLUMNS = ['model', 'run', 'method', 'dataset', 'split', 'trained_on', 'backbone', 'seed', 'balance', 'lr', 'tau',
                'batch_size', 'best_epoch', 'n_trainable_parameters', 'n_queries']

def build_zero_shot(name, args, device):
    method_name, kwargs = zero_shot[name]
    dims = BACKBONE_DIMS[args.backbone]

    if method_name == 'llm':
        #TODO: the path of the classifications file and the Hugging Face cache folder have no source since the arguments were removed
        raise NotImplementedError("The llm method needs the path of the classifications file and of the Hugging Face cache")

    #loss_tau is required by the constructor, but a zero-shot method has no loss to train
    method = methods[method_name](loss_tau=1.0, global_dim=dims['global_tokens'], **kwargs)
    method.to(device)

    info = dict.fromkeys(INFO_COLUMNS)
    info.update(model=name, run=name, method=method_name, backbone=args.backbone)

    return info, method

def build_trained(run, args, device):
    run_dir = os.path.join(RUNS_FOLDER, run)
    with open(os.path.join(run_dir, "settings.json")) as f:
        settings = json.load(f)

    dims = BACKBONE_DIMS[settings['backbone']]
    method = methods[settings['model']](
        loss_tau=settings['tau'],
        global_dim=dims['global_tokens'],
        audio_local_dim=dims['local_audio_tokens'],
        text_local_dim=dims['local_text_tokens']
    )
    method.to(device)
    method.load_checkpoint(run_dir, args.checkpoint)

    balance_str = "balanced" if settings['balance'] else "unbalanced"

    info = dict.fromkeys(INFO_COLUMNS)
    info.update(model=f"{settings['model']}_{settings['seed']}_{balance_str}", run=run, method=settings['model'],
                trained_on=settings['dataset'], backbone=settings['backbone'], seed=settings['seed'], balance=settings['balance'],
                lr=settings['lr'], tau=settings['tau'], batch_size=settings['batch_size'], best_epoch=method.epoch + 1,
                n_trainable_parameters=settings['n_trainable_parameters'])

    return info, method

def find_runs(args):
    #the runs of the dataset with a settings.json and the checkpoint to evaluate
    runs = args.runs if args.runs is not None else sorted(os.listdir(RUNS_FOLDER))

    selected = []
    for run in runs:
        run_dir = os.path.join(RUNS_FOLDER, run)
        settings_path = os.path.join(run_dir, "settings.json")

        if not os.path.exists(settings_path) or not os.path.exists(os.path.join(run_dir, f"{args.checkpoint}.pth.tar")):
            print(f"Skipping {run}: no settings.json or no {args.checkpoint} checkpoint")
            continue

        with open(settings_path) as f:
            if json.load(f)['dataset'].lower() != args.dataset.lower():
                print(f"Skipping {run}: trained on a different dataset")
                continue

        selected.append(run)

    return selected

def evaluate_run(method, eval_set, relevance_fns, ks=KS):
    """
    relevance_fns: dict {criterion name: function of the eval_set that returns the relevant positions of each triplet}
    Returns {criterion name: {'hit@1': ..., 'ndcg@100': ...}} with the mean of every metric over the triplets, as plain numbers.
    A criterion where some triplet has no relevant track has no scores (None), because there is no policy for those triplets.
    """
    results = {}

    relevant_positions = {}
    for name, relevance_fn in relevance_fns.items():
        positions = relevance_fn(eval_set)

        relevant_positions[name] = positions

    if len(relevant_positions) == 0:
        return results

    run = Retriever(method, eval_set, top_k=max(ks)).retrieve()     #ranked lists, once for all the criteria

    for name, positions in relevant_positions.items():
        is_relevant, num_relevant = build_relevance(run, positions)
        per_query_scores = score_run(is_relevant, num_relevant, ks)
        results[name] = aggregate(per_query_scores)

    return results

def main():
    #evaluates the zero-shot methods and the trained runs. One file per relevance criterion, one row per model, one column per metric
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    relevance_fns = {name: CRITERIA[name] for name in args.relevance}

    #one evaluation set for each backbone and kind of input: global embeddings only, or global and local tokens
    eval_sets = {}
    def get_eval_set(backbone, return_local_tokens):
        if (backbone, return_local_tokens) not in eval_sets:
            eval_sets[(backbone, return_local_tokens)] = EvaluationSet(args.dataset, args.split, return_local_tokens, backbone)
        return eval_sets[(backbone, return_local_tokens)]

    #the methods are built one at a time, so only one is in memory
    def iterate_methods():
        for name in args.zero_shot:
            yield build_zero_shot(name, args, device)
        for run in find_runs(args):
            yield build_trained(run, args, device)

    for info, method in iterate_methods():
        print(f"\nEvaluating {info['model']} ({info['run']})")

        eval_set = get_eval_set(info['backbone'], method.needs_local_tokens)
        info.update(dataset=args.dataset.lower(), split=args.split, n_queries=len(eval_set))

        results = evaluate_run(method, eval_set, relevance_fns)

        for criterion, scores in results.items():
            if scores is None:
                continue

            csv_path = os.path.join(RESULTS_FOLDER, f"{args.dataset.lower()}_{args.split}_{criterion}.csv")
            save_results_csv(csv_path, [{**info, **scores}])
            print(f"{criterion}: map@10 {scores['map@10']:.4f}  hit@10 {scores['hit@10']:.4f}  -> {csv_path}")


if __name__ == "__main__":
    main()


# class ModelEvaluationMTIQ(ModelEvaluationBase):
#     def __init__(self, model_weights_path, model_architecture, save_path, split):
#         self.triplets_df = pd.read_csv(f'{TRIPLETS_FOLDER}/MTIQ/{split}_triplets.csv')

#         similarity_df = pd.read_csv(f'path/to/similarity_mtiq_{split}.csv')
#         self.similarity_df = similarity_df.set_index('reference_song')  # Set 'reference_song' as the index for easy lookup 
#         self.similarity_df['similar_songs'] = self.similarity_df['similar_songs'].apply(ast.literal_eval) 

#         #merge tags_df with splits_df to get only the songs in the test set
#         splits_df = pd.read_csv('path/to/splits.csv')
#         test_split = splits_df[splits_df['split'] == split]

#         tags_df = pd.read_csv('path/to/labels.csv')
#         tags_df = tags_df.merge(test_split[['song']], on='song', how='inner')
#         tags_df['song'] = 'original/' + tags_df['song'] + '.mp3' #so the idx are in the 'save name'
#         self.tags_df = tags_df.set_index('song')  # Set 'song' as the index for easy lookup

#         self.all_songs = self.tags_df.index.tolist()

#         super().__init__(model_weights_path, model_architecture, save_path, 'mtiq')
    
#     def evaluate(self):
#         RaK_by_type = {}
#         for key in self.triplets_df['modification'].unique().tolist():
#             RaK_by_type[key] = {'file': {k: 0 for k in self.k_values}, 
#                                 'tag': {k: 0 for k in self.k_values}, 
#                                 'similarity': {k: 0 for k in self.k_values},
#                                 'both': {k: 0 for k in self.k_values}}
        
#         RaK_by_type['global'] = {'file': {k: 0 for k in self.k_values}, 
#                                 'tag': {k: 0 for k in self.k_values}, 
#                                 'similarity': {k: 0 for k in self.k_values},
#                                 'both': {k: 0 for k in self.k_values}}

#         #----------------------------------------------evaluate---------------------------------------------------------#    
#         #for each song, i get their recall at k
#         for i, (index, row) in enumerate(self.triplets_df.iterrows()):
#             if i % 10 == 0:
#                 print(f"{i}/{len(self.triplets_df)}", end='\r')

#             #get composed embedding
#             reference_song_id = row['reference_song'].split('.')[0]
#             composed_embedding = self.get_composed_embedding(row)

#             #get ground truth
#             ground_truth = row['target_song'].split('.')[0]

#             #get similar songs to the fusioned embedding
#             distances, indices = self.knn.kneighbors(composed_embedding.reshape(1, -1))
#             filtered_indices = [i for i in indices[0] if i != self.song_to_position[reference_song_id]] # Exclude the reference song itself
#             similar_songs = [self.position_to_song[position] for position in filtered_indices]

#             #get similar songs to the reference song
#             similar_songs_to_ref = self.similarity_df.loc[reference_song_id + '.mp3']['similar_songs']
#             similar_songs_to_ref = [song.split('.')[0] for song in similar_songs_to_ref]

#             #ground truth tag
#             modification = row['modification']
#             ground_truth_tag = modifications_reverse[modification]

#             #for each k, i get their recall at k
#             for k in self.k_values:
#                 #FILE RECALL @ K
#                 #if the ground truth in predicted in the top k predictions, i return 1, else 0
#                 if ground_truth in similar_songs[:k]:
#                     recall = 1
#                 else:
#                     recall = 0
                
#                 RaK_by_type[modification]['file'][k] += recall
#                 RaK_by_type['global']['file'][k] += recall

#                 #SIMILARITY RECALL @ K
#                 #check if at least one of the top k predictions is a similar song to the reference
#                 if any(song in similar_songs[:k] for song in similar_songs_to_ref):
#                     recall = 1
#                 else:
#                     recall = 0

#                 RaK_by_type[modification]['similarity'][k] += recall
#                 RaK_by_type['global']['similarity'][k] += recall
            
#                 #TAG RECALL @ K
#                 #if the ground truth tag appears in the top k predictions, I return 1, else 0
#                 #get tags of top k predictions
#                 top_k_tags = set()
#                 for song in similar_songs[:k]:
#                     tags = self.tags_df.loc[song+'.mp3']
#                     top_k_tags.update(tag for tag in tags if tag != 'uncertain')
#                 if ground_truth_tag in top_k_tags:
#                     recall = 1
#                 else:
#                     recall = 0
#                 RaK_by_type[modification]['tag'][k] += recall
#                 RaK_by_type['global']['tag'][k] += recall

#                 #BOTH RECALL @ K
#                 #if at least one song is similar, and the tag is correct, i return 1, else 0
                
#                 # get matching similar songs within top-k
#                 matched_songs = [song for song in similar_songs_to_ref if song in similar_songs[:k]]

#                 if matched_songs:  # at least one match exists
#                     matched_tags = set()
#                     for song in matched_songs:
#                         tags = self.tags_df.loc[song+'.mp3']
#                         matched_tags.update(tag for tag in tags if tag != 'uncertain')
#                     if ground_truth_tag in matched_tags: #at least one of the matched songs has the correct tag
#                         recall = 1
#                     else:
#                         recall = 0
#                 else:
#                     recall = 0

#                 RaK_by_type[modification]['both'][k] += recall
#                 RaK_by_type['global']['both'][k] += recall


#         for modification in RaK_by_type.keys():
#             print(f"Recall@K for modification '{modification}':")

#             if modification == 'global':
#                 n_triplets = len(self.triplets_df)
#             else:
#                 n_triplets = len(self.triplets_df[self.triplets_df['modification'] == modification])

#             #normalizando RaK
#             RaK_by_type[modification]['file'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['file'].items()}
#             RaK_by_type[modification]['tag'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['tag'].items()}
#             RaK_by_type[modification]['similarity'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['similarity'].items()}
#             RaK_by_type[modification]['both'] = {k: v / n_triplets for k, v in RaK_by_type[modification]['both'].items()}
#             rak_types = ['file', 'tag', 'similarity', 'both']
#             raks = [RaK_by_type[modification][rak_type] for rak_type in rak_types]

#             #salvando resultados no csv
#             csv_path = os.path.join(self.save_path, f"results_{modification}.csv")
#             self.save_results_csv(csv_path, self.model_architecture, self.model_weights_path, self.k_values, raks, rak_types)