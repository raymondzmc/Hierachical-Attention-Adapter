import os
import json
import nltk
import argparse
import evaluate
import numpy as np
from bart_score import BARTScorer


import pdb

# nltk.download('punkt')


def main(args):
    prediction_file = os.path.join(args.checkpoint_dir, 'prediction_results.json')
    if not os.path.isfile(prediction_file):
        raise FileNotFoundError(f"Cannot find \"prediction_results.json\" in {args.checkpoint_dir}")
    
    predictions = json.load(open(prediction_file))
    rouge = evaluate.load('rouge')
    # bartscore = evaluate.load("bartscore", model_type="distilbert-base-uncased")
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    
    
    sent_len = [len(nltk.sent_tokenize(pred['summary'])) for pred in predictions]
    reference_sentences = ['\n'.join(nltk.sent_tokenize(pred['summary'])) for pred in predictions]
    pred_sentences = ['\n'.join(nltk.sent_tokenize(pred['prediction'])[:sent_len[i]]) for i, pred in enumerate(predictions)]
    bart_score = bart_scorer.score(reference_sentences, pred_sentences, batch_size=16)
        # pdb.set_trace()
    print(rouge.compute(predictions=pred_sentences, references=reference_sentences))
    # precision, recall, f1 = 0, 0, 0
    # for i, pred in enumerate(predictions):
    #     reference_sentences = nltk.sent_tokenize(pred['summary'])
    #     pred_sentences = nltk.sent_tokenize(pred['prediction'])[:sent_len[i]]
    #     try:
    #         bart_scores = bartscore.compute(predictions=pred_sentences, references=reference_sentences, lang="en")
    #     except:
    #         pdb.set_trace()
    #     precision += np.mean(bart_scores['precision'])
    #     recall += np.mean(bart_scores['recall'])
    #     f1 += np.mean(bart_scores['f1'])
    print(f"[BARTScore] {np.mean(bart_score)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='evaluatePredictions', description='Evaluate the prediction files from Mixtral output')
    parser.add_argument('--checkpoint_dir', type=str, required=True, default=None)
    parser.add_argument('--num_sentences', type=int, default=4)
    args = parser.parse_args()

    main(args)