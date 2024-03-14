import datasets
import pdb

def get_preprocessed_mediasum(tokenizer, return_token_type_ids=False):
    dataset = datasets.load_dataset("ccdv/mediasum", trust_remote_code=True)
    pdb.set_trace()
    
    # Default config uses RoBERTa seperator "</s>"
    turn_separator = '</s>'

    def preprocess_function(example, split=None):
        if split != None:
            dataset = dataset[split]
        
        summary = "Summary: " + example["summary"] + tokenizer.eos_token
        summary_ids = tokenizer.encode(summary, add_special_tokens=False)
        
        dialogue_turns = example['document'].split(turn_separator)
        if return_token_type_ids:
            dialogue_turns = [turn + '\n' for turn in dialogue_turns]
            dialogue_turns = example["dialogue"].split(turn_separator)
            encoded_dialogue_turns = [tokenizer.encode(turn.strip(), add_special_tokens=False) for turn in dialogue_turns]
            dialogue_ids = [tokenizer.bos_token_id]
            token_type_ids = [0]
            for i, ids in enumerate(encoded_dialogue_turns):
                turn_id = i + 1 # 0 is used as pad token
                dialogue_ids.extend(ids + [13])
                token_type_ids.extend(len(ids) * [turn_id] + [0])
            token_type_ids[-1] = -1
            sample = {
                "input_ids": dialogue_ids + summary_ids,
                "attention_mask": [1] * (len(dialogue_ids) + len(summary_ids)),
                "labels": [-100] * len(dialogue_ids) + summary_ids,
                "token_type_ids": token_type_ids + [-1] * len(summary_ids),
            }
        else:
            dialogue = '\n'.join(dialogue_turns)
            dialogue_ids = tokenizer.encode(tokenizer.bos_token + dialogue,
                                            add_special_tokens=False)
            sample = {
                "input_ids": dialogue_ids + summary_ids,
                "attention_mask": [1] * (len(dialogue_ids) + len(summary_ids)),
                "labels": [-100] * len(dialogue_ids) + summary_ids,
            }

        return sample
    
    dataset = dataset.filter(lambda x: len(x["document"].split(turn_separator)) > 1)
    dataset = dataset.map(preprocess_function, num_proc=64)

    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = get_preprocessed_mediasum(tokenizer)
    print(f"Number of train examples: {len(dataset['train'])}")
    print(f"Number of train examples: {len(dataset['train'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['train']])}.")
    print(f"Number of validation examples: {len(dataset['validation'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['validation']])}.")
    print(f"Number of test examples: {len(dataset['test'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['test']])}.")