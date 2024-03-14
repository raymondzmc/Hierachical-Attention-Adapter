import pdb
import os
import json
import numpy as np
from datasets import DatasetDict, Dataset

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'FriendsQA')

def load_friendsqa_data(data_path):
    file_paths = {'train': os.path.join(data_path, 'friendsqa_trn.json'),
                  'validation': os.path.join(data_path, 'friendsqa_dev.json'),
                  'test': os.path.join(data_path, 'friendsqa_tst.json')}
    data = {k: {'dialogue': [], 'question': [], 'answers': []} for k in file_paths.keys()}
    for split, file_path in file_paths.items():
        split_data = json.load(open(file_path))
        for example in split_data['data']:
            assert len(example['paragraphs']) == 1, f"Example {example['title']} has more than one paragraphs!"
            paragraphs = example['paragraphs'][0]
            dialogue = [f"{', '.join(u['speakers'])}: {u['utterance']}" if u['speakers'][0] != '#NOTE#' else u['utterance'] 
                        for u in paragraphs['utterances:']]
            for question in paragraphs['qas']:
                data[split]['dialogue'].append(dialogue)
                data[split]['question'].append(question['question'])
                data[split]['answers'].append([ans['answer_text'] for ans in question['answers']])
    return data

def get_preprocessed_friendsqa(tokenizer, return_token_type_ids=False):
    
    data = load_friendsqa_data(DATA_PATH)
    dataset = DatasetDict({k: Dataset.from_dict(v) for k, v in data.items()})
    def preprocess_function(example):
        
        # Use the first answer
        answer = example['answers'][0]
        answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
        question_prompt = f"Question: {example['question']}\nAnswer: "

        if return_token_type_ids:
            dialogue_token_ids = [tokenizer.bos_token_id]
            token_type_ids = [0]
            for idx, turn in enumerate(example["dialogue"]):
                turn_id = idx + 1 # 0 is used as pad token
                dialogue_turn = turn.strip() + '\n'
                turn_token_ids = tokenizer.encode(dialogue_turn, add_special_tokens=False)
                dialogue_token_ids.extend(turn_token_ids)
                token_type_ids.extend(len(turn_token_ids) * [turn_id])

            # Add summary prompt to the end of input ids
            question_prompt_ids = tokenizer.encode(question_prompt, add_special_tokens=False)
            dialogue_token_ids.extend(question_prompt_ids)
            token_type_ids.extend(len(question_prompt_ids) * [-1])
            token_type_ids = token_type_ids + [-1] * len(answer_token_ids)
            token_type_ids = [np.int32(x) for x in token_type_ids] 
            sample = {
                "input_ids": dialogue_token_ids + answer_token_ids,
                "attention_mask": [1] * (len(dialogue_token_ids) + len(answer_token_ids)),
                "labels": [-100] * len(dialogue_token_ids) + answer_token_ids,
                "token_type_ids": token_type_ids,
            }
        else:
            dialogue = tokenizer.bos_token + "\n".join(example["dialogue"]) + '\n' + question_prompt
            dialogue_token_ids = tokenizer.encode(dialogue, add_special_tokens=False)
            
            sample = {
                "input_ids": dialogue_token_ids + answer_token_ids,
                "attention_mask": [1] * (len(dialogue_token_ids) + len(answer_token_ids)),
                "labels": [-100] * len(dialogue_token_ids) + answer_token_ids,
            }
        return sample

    # dataset['train'] = dataset.map(preprocess_function, num_proc=16)
    # dataset['validation'] = dataset.map(preprocess_function, num_proc=16)
    dataset = dataset.map(preprocess_function, num_proc=16)
    
    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = get_preprocessed_friendsqa(tokenizer)
    print(f"Number of train examples: {len(dataset['train'])}")
    print(f"Number of train examples: {len(dataset['train'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['train']])}.")
    print(f"Number of validation examples: {len(dataset['validation'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['validation']])}.")
    print(f"Number of test examples: {len(dataset['test'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['test']])}.")