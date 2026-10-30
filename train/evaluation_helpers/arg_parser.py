import argparse

from evaluation_helpers.relevance import CRITERIA
from utils.backbones import BACKBONE_DIMS

ZERO_SHOT_CHOICES = ['audio_only', 'text_only', 'sum', 'llm']

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Configuration")
    
    parser.add_argument('--dataset', type=str, default='mtsq', help='Dataset Name')
    parser.add_argument('--split', type=str, default='test', help='Evaluation Split (test or val)')
    parser.add_argument('--checkpoint', type=str, default='model_best', help="Checkpoint of the runs: 'model_best' or 'checkpoint' (last epoch)")

    parser.add_argument('--backbone', type=str, default='laion_clap', choices=list(BACKBONE_DIMS), help='Backbone of the zero-shot methods (the trained runs use the one in their settings.json)')

    parser.add_argument('--runs',type=str, nargs='*', default=None, help='Names of the run folders to evaluate. Default: every run of the dataset in runs_dir')
    parser.add_argument('--zero_shot', type=str, nargs='*', default=['audio_only', 'text_only', 'sum'], choices=ZERO_SHOT_CHOICES, help='Zero-shot methods to evaluate')

    parser.add_argument('--relevance', type=str, nargs='+', default=['exact'], choices=list(CRITERIA), help='Relevance criteria, one results file for each')

    args = parser.parse_args()

    return args
