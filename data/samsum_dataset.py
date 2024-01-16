import copy
import datasets


def get_preprocessed_samsum(tokenizer):
    dataset = datasets.load_dataset("samsum")

    # prompt = (
    #     f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    # )

    # def apply_prompt_template(sample):
    #     return {
    #         # "prompt": prompt.format(dialog=sample["dialogue"]),
    #         "dialogue": sample["dialogue"],
    #         "summary": sample["summary"],
    #     }

    def preprocess_function(example, split=None):
        if split != None:
            dataset = dataset[split]

        dialogue = tokenizer.encode(tokenizer.bos_token + example["dialogue"],
                                    add_special_tokens=False)
        summary = tokenizer.encode(example["summary"] + tokenizer.eos_token,
                                   add_special_tokens=False)

        sample = {
            "input_ids": dialogue + summary,
            "attention_mask": [1] * (len(dialogue) + len(summary)),
            "labels": [-100] * len(dialogue) + summary,
        }

        return sample

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