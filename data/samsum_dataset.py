import copy
import datasets
import pdb

def get_preprocessed_samsum(tokenizer, return_token_type_ids=False):
    data_files = {'train': 'resources/SamSum/train.json',
                  'validation': 'resources/SamSum/validation.json',
                  'test': 'resources/SamSum/test.json'}
    dataset = datasets.load_dataset("json", data_files=data_files)
    turn_separator = "\n"
    def preprocess_function(example, split=None):
        if split != None:
            dataset = dataset[split]

        summary_ids = tokenizer.encode(example["summary"] + tokenizer.eos_token, add_special_tokens=False)
        
        if return_token_type_ids:
            dialogue_turns = [(turn.strip() + "\n") for turn in example["dialogue"].split(turn_separator)]
            encoded_dialogue_turns = [tokenizer.encode(turn, add_special_tokens=False) for turn in dialogue_turns]
            dialogue_ids = [tokenizer.bos_token_id]
            token_type_ids = [0]
            for i, ids in enumerate(encoded_dialogue_turns):
                turn_id = i + 1 # 0 is used as pad token
                dialogue_ids.extend(ids)
                token_type_ids.extend(len(ids) * [turn_id])
            
            # Add summary prompt to the end of input ids
            summary_prompt_ids = tokenizer.encode("Summary: ", add_special_tokens=False)
            dialogue_ids.extend(summary_prompt_ids)
            token_type_ids.extend(len(summary_prompt_ids) * [-1])

            sample = {
                "input_ids": dialogue_ids + summary_ids,
                "attention_mask": [1] * (len(dialogue_ids) + len(summary_ids)),
                "labels": [-100] * len(dialogue_ids) + summary_ids,
                "token_type_ids": token_type_ids + [-1] * len(summary_ids),
            }
        else:
            dialogue_ids = tokenizer.encode(tokenizer.bos_token + example["dialogue"] + '\n',
                                            add_special_tokens=False)
            sample = {
                "input_ids": dialogue_ids + summary_ids,
                "attention_mask": [1] * (len(dialogue_ids) + len(summary_ids)),
                "labels": [-100] * len(dialogue_ids) + summary_ids,
            }

        return sample
    
    dataset = dataset.filter(lambda x: len(x["dialogue"].split(turn_separator)) > 1)
    dataset = dataset.map(preprocess_function)

    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = get_preprocessed_samsum(tokenizer)
    print(f"Number of train examples: {len(dataset['train'])}")
    print(f"Number of train examples: {len(dataset['train'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['train']])}.")
    print(f"Number of validation examples: {len(dataset['validation'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['validation']])}.")
    print(f"Number of test examples: {len(dataset['test'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['test']])}.")