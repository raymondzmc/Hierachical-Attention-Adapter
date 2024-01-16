import pdb
import datasets


def get_preprocessed_summscreen(tokenizer):
    dataset = datasets.load_dataset("YuanPJ/summ_screen", "tms")
    def preprocess_function(example):
        dialogue = tokenizer.bos_token + "\n".join(example["Transcript"]) + "\n"
        summary = " ".join(example["Recap"]) + tokenizer.eos_token
        dialogue_tokens = tokenizer.encode(dialogue, add_special_tokens=False)
        summary_tokens = tokenizer.encode(summary, add_special_tokens=False)
        sample = {
            "input_ids": dialogue_tokens + summary_tokens,
            "attention_mask": [1] * (len(dialogue_tokens) + len(summary_tokens)),
            "labels": [-100] * len(dialogue_tokens) + summary_tokens,
        }
        return sample

    def eval_preprocess_function(example):
        dialogue = tokenizer.bos_token + "\n".join(example["Transcript"]) + "\n"
        summary = " ".join(example["Recap"]) + tokenizer.eos_token
        dialogue_tokens = tokenizer.encode(dialogue, add_special_tokens=False)
        summary_tokens = tokenizer.encode(summary, add_special_tokens=False)
        sample = {
            "input_ids": dialogue_tokens + summary_tokens,
            "attention_mask": [1] * (len(dialogue_tokens) + len(summary_tokens)),
            "labels": [-100] * len(dialogue_tokens) + summary_tokens,
        }
        return sample

    # dataset['train'] = dataset.map(preprocess_function, num_proc=16)
    # dataset['validation'] = dataset.map(preprocess_function, num_proc=16)
    dataset = dataset.map(preprocess_function, num_proc=16)
    
    return dataset

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = get_preprocessed_summscreen(tokenizer)
    print(f"Number of train examples: {len(dataset['train'])}")
    print(f"Number of train examples: {len(dataset['train'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['train']])}.")
    print(f"Number of validation examples: {len(dataset['validation'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['validation']])}.")
    print(f"Number of test examples: {len(dataset['test'])}, max sequence length: {max([len(x['input_ids']) for x in dataset['test']])}.")