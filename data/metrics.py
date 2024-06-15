import nltk
import numpy as np
import re
import string
from collections import Counter
from rouge_score import rouge_scorer, scoring
from openai import OpenAI
from mosestokenizer import *
from collections import defaultdict
from tqdm import tqdm

nltk.download('stopwords')
secret_key = "sk-proj-e6I9bEgzGsEU2se35HAqT3BlbkFJpSZwRCMY9JjAnioYCgpp"

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2',  'rougeL', 'rougeLsum'], use_stemmer=True)

def rouge_metrics(references: list[str], predictions: list[str]):
    references = ["\n".join(nltk.sent_tokenize(ref.strip())) for ref in references]
    predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    aggregator = scoring.BootstrapAggregator(n_samples=10000)
    result = {k: [] for k in scorer.rouge_types}
    for ref_text, pred_text in zip(references, predictions):
        score = scorer.score(ref_text, pred_text)
        aggregator.add_scores(score)
    result = aggregator.aggregate()
    for key in result:
        result[key] = round(result[key].mid.fmeasure, 4)
    return result

def gpt_score_metrics(references: list[str], predictions: list[str], model: str = 'davinci-002'):

    ASPECT_DEFINITIONS =  {
        "informativeness": "Convert the following text into another expression that preserves key information:",
        "naturalness": "Convert the following text into another expression that is as natural as a native speaker:",
        "quality": "Convert the following text into another expression that is fluent and grammatically correct:",
    }

    def add_dot(text):
        if text.strip()[-1] != '.':
            text = text.strip() + ' .'
        new_text = text
        return new_text

    detokenizer = MosesDetokenizer('en')

    client = OpenAI(api_key=secret_key, organization='org-OgCRLOoWI3PvTp2e4wrID5bc', project='proj_JJ9XVpEKu69t8T44bomdBb7s')
    
    aspect_scores = {aspect: defaultdict(list) for aspect in ASPECT_DEFINITIONS.keys()}
    for ref_summ, pred_summ in tqdm(zip(references, predictions), desc="Computing GPTScore", total=len(references)):
        # ref_summ = add_dot(detokenizer(ref_text.split(" ")))
        # pred_summ = add_dot(detokenizer(pred_text.split(" ")))
        # import pdb; pdb.set_trace()
        
        for aspect, definition in ASPECT_DEFINITIONS.items():
            prompts = {'ref_hypo': (f"{definition}\n{ref_summ} In other words , \n", pred_summ),
                       'hypo_ref': (f"{definition}\n{pred_summ} In other words , \n", ref_summ)}
            scores = defaultdict(None)
            for score_type, (input_prompt, output_prompt) in prompts.items():
                prompt = input_prompt + output_prompt
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=0,
                    temperature=0,
                    logprobs=0,
                    top_p=1,
                    frequency_penalty=0,
                    echo=True,
                    n=None,
                )
                logprobs = response.choices[0].logprobs
                idx_start = logprobs.text_offset.index(len(input_prompt))
                loss = -sum(logprobs.token_logprobs[idx_start:-1])
                avg_loss = loss / (len(logprobs.tokens[idx_start:-1]))
                score = -avg_loss
                scores[score_type] = score
            aspect_scores[aspect]['ref_hypo'].append(scores['ref_hypo'])
            aspect_scores[aspect]['hypo_ref'].append(scores['hypo_ref'])
            aspect_scores[aspect]['avg_f1'].append(0.5 * (scores['hypo_ref'] + scores['ref_hypo']))
            aspect_scores[aspect]['harm_f'].append(scores['hypo_ref'] * scores['ref_hypo'] /\
                                                   (scores['hypo_ref'] + scores['ref_hypo']))

    aggregate_scores = {}
    for aspect, _scores in aspect_scores.items():
        aggregate_scores[aspect] = {k: np.mean(v) for k, v in _scores.items()}
    return aggregate_scores

def summarization_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    aggregator = scoring.BootstrapAggregator(n_samples=10000)
    result = {k: [] for k in scorer.rouge_types}
    for label, pred in zip(decoded_labels, decoded_preds):
        score = scorer.score(label, pred)
        aggregator.add_scores(score)
    result = aggregator.aggregate()
    for key in result:
        result[key] = result[key].mid.fmeasure
    return result

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_underline(text):
        return text.replace('_', ' ')

    return remove_underline(white_space_fix(remove_articles(remove_punc(lower(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def qa_metrics(tokenizer, eval_preds):
    preds, labels = eval_preds
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    f1_scores = [f1_score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    exact_match_scores = [exact_match_score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    return {'f1': 100 * np.mean(f1_scores),
            'exact_match': 100 * np.mean(exact_match_scores)}


def get_summarization_metrics(tokenizer):
    return lambda pred: summarization_metrics(tokenizer, pred)

def get_qa_metrics(tokenizer):
    return lambda pred: qa_metrics(tokenizer, pred)