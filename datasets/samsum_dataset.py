import copy
import datasets


def get_preprocessed_samsum(tokenizer, split):
    dataset = datasets.load_dataset("samsum", split=split)

    # prompt = (
    #     f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    # )

    # def apply_prompt_template(sample):
    #     return {
    #         # "prompt": prompt.format(dialog=sample["dialogue"]),
    #         "dialogue": sample["dialogue"],
    #         "summary": sample["summary"],
    #     }

    def preprocess_function(example):
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

    dataset = dataset.map(preprocess_function,
                          remove_columns=list(dataset.features))

    return dataset
