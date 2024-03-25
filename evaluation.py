import os
import json
import nltk
import argparse
import evaluate
import numpy as np
from bart_score import BARTScorer
from rouge_score import rouge_scorer


import pdb

# nltk.download('punkt')


def main(args):
    prediction_file = os.path.join(args.checkpoint_dir, 'prediction_results.json')
    if not os.path.isfile(prediction_file):
        raise FileNotFoundError(f"Cannot find \"prediction_results.json\" in {args.checkpoint_dir}")
    
    predictions = json.load(open(prediction_file))
    if args.subset is not None:
        predictions = predictions[:args.subset]
    
    # bartscore = evaluate.load("bartscore", model_type="distilbert-base-uncased")
    rouge = evaluate.load('rouge')
    bart= BARTScorer(device='cuda:1', checkpoint='facebook/bart-large-cnn')
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2',  'rougeL'], use_stemmer=False)
    
    # sent_len = [len(nltk.sent_tokenize(pred['summary'])) for pred in predictions]
    reference_sentences = ['\n'.join(nltk.sent_tokenize(pred['summary'])) for i, pred in enumerate(predictions)]
    pred_sentences = ['\n'.join(nltk.sent_tokenize(pred['prediction'])) for i, pred in enumerate(predictions)]
    bart_score = bart.score(reference_sentences, pred_sentences, batch_size=16)
    print(f"[BARTScore] {np.mean(bart_score)}")
        # pdb.set_trace()
    rouge_scores = {k: [] for k in rouge.rouge_types}
    for ref, pred in zip(reference_sentences, pred_sentences):
        score = rouge.score(ref, pred)
        for k, v in score.items():
            rouge_scores[k].append(v.fmeasure)
    for key in rouge_scores:
        rouge_scores[key] = np.mean(rouge_scores[key])
        print(f"{key}: {rouge_scores[key]}")
    print(f"[ROUGE Average]: {np.mean(list(rouge_scores.values()))}")
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='evaluatePredictions', description='Evaluate the prediction files from Mixtral output')
    parser.add_argument('--checkpoint_dir', type=str, required=True, default=None)
    parser.add_argument('--num_sentences', type=int, default=None)
    parser.add_argument('--subset', type=int, default=None)
    args = parser.parse_args()

    main(args)