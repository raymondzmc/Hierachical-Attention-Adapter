import pdb
import datasets


def get_preprocessed_summscreen(tokenizer, return_token_type_ids=False, test_only=False):
    split = 'test' if test_only else None
    dataset = datasets.load_dataset("YuanPJ/summ_screen", "tms", split=split)
    def preprocess_function(example):
        summary_token_ids = tokenizer.encode(" ".join(example['Recap']), add_special_tokens=False)
        if return_token_type_ids:
            dialogue_token_ids = [tokenizer.bos_token_id]
            token_type_ids = [0]
            for idx, turn in enumerate(example["Transcript"]):
                turn_id = idx + 1 # 0 is used as pad token
                dialogue_turn = turn.strip() + '\n'
                turn_token_ids = tokenizer.encode(dialogue_turn, add_special_tokens=False)
                dialogue_token_ids.extend(turn_token_ids)
                token_type_ids.extend(len(turn_token_ids) * [turn_id])

            # Add summary prompt to the end of input ids
            summary_prompt_ids = tokenizer.encode("Recap: ", add_special_tokens=False)
            dialogue_token_ids.extend(summary_prompt_ids)
            token_type_ids.extend(len(summary_prompt_ids) * [-1])
            sample = {
                "id": example['File Name'],
                "input_ids": dialogue_token_ids + summary_token_ids,
                "attention_mask": [1] * (len(dialogue_token_ids) + len(summary_token_ids)),
                "labels": [-100] * len(dialogue_token_ids) + summary_token_ids,
                "token_type_ids": token_type_ids + [-1] * len(summary_token_ids),
            }
            
        else:
            dialogue = tokenizer.bos_token + "\n".join(example["Transcript"]) + "\nSummary:"
            dialogue_token_ids = tokenizer.encode(dialogue, add_special_tokens=False)
            sample = {
                "id": example['File Name'],
                "input_ids": dialogue_token_ids + summary_token_ids,
                "attention_mask": [1] * (len(dialogue_token_ids) + len(summary_token_ids)),
                "labels": [-100] * len(dialogue_token_ids) + summary_token_ids,
            }
        return sample

    # dataset['train'] = dataset.map(preprocess_function, num_proc=16)
    # dataset['validation'] = dataset.map(preprocess_function, num_proc=16)
    dataset = dataset.map(preprocess_function, num_proc=32)
    
    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = get_preprocessed_summscreen(tokenizer, return_token_type_ids=True)
    print(f"Number of train examples: {len(dataset['train'])}")
    print(f"Number of train examples: {len(dataset['train'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['train']])}.")
    print(f"Number of validation examples: {len(dataset['validation'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['validation']])}.")
    print(f"Number of test examples: {len(dataset['test'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['test']])}.")