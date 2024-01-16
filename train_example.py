import gc
import os
import sys
import json
import threading

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model
from data import get_preprocessed_samsum

logger = get_logger(__name__, log_level="INFO")

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

def main():
    accelerator = Accelerator()
    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    dataset_name = "samsum_dataset"
    text_column = "dialogue"
    label_column = "summary"
    lr = 1e-4
    num_epochs = 3
    batch_size = 2
    gradient_accumulation_steps = 4 # effective batch size = batch_size * gradient_accumulation_steps
    seed = 42
    max_length = 2048
    do_test = True
    use_peft = True
    set_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    with accelerator.main_process_first():
        train_dataset = get_preprocessed_samsum(tokenizer, 'train')
        eval_dataset = get_preprocessed_samsum(tokenizer, 'validation')
        test_dataset = get_preprocessed_samsum(tokenizer, 'test')
    accelerator.wait_for_everyone()
    collate_fn = DataCollatorForSeq2Seq(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )

    print(next(iter(train_dataloader)))
    
    # creating model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype = torch.bfloat16)
    
    if use_peft:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    is_ds_zero_3 = True
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

    for epoch in range(num_epochs):
        # with TorchTracemalloc() as tracemalloc:
        #     model.train()
        #     total_loss = 0
        #     for step, batch in enumerate(tqdm(train_dataloader)):
        #         outputs = model(**batch)
        #         loss = outputs.loss
        #         loss = loss / gradient_accumulation_steps
        #         total_loss += loss.detach().float()
        #         accelerator.backward(loss)
        #         if step % gradient_accumulation_steps == 0:
        #             optimizer.step()
        #             lr_scheduler.step()
        #             optimizer.zero_grad()
        # # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        # accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        # accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        # accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        # accelerator.print(
        #     "GPU Total Peak Memory consumed during the train (max): {}".format(
        #         tracemalloc.peaked + b2mb(tracemalloc.begin)
        #     )
        # )

        # accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        # accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        # accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        # accelerator.print(
        #     "CPU Total Peak Memory consumed during the train (max): {}".format(
        #         tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
        #     )
        # )
        # train_epoch_loss = total_loss / len(train_dataloader)
        # train_ppl = torch.exp(train_epoch_loss)
        # accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")
        
        # accelerator.wait_for_everyone()
        
        save_id = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_epoch{epoch}".replace(
            "/", "_"
        )
        # unwrapped_model = accelerator.unwrap_model(model)
        # unwrapped_model.save_pretrained(
        #     f'./output/{save_id}',
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=accelerator.get_state_dict(model),
        # )

        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch,
                        max_new_tokens=4096,
                        synced_gpus=is_ds_zero_3,
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        with open(f"./output/{save_id}/eval_results.json") as f:
            json.dump(eval_preds, f, indent=4) 
        
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        # correct = 0
        # total = 0
        # assert len(eval_preds) == len(
        #     dataset["train"][label_column]
        # ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
        # for pred, true in zip(eval_preds, dataset["train"][label_column]):
        #     if pred.strip() == true.strip():
        #         correct += 1
        #     total += 1
        # accuracy = correct / total * 100
        # accelerator.print(f"{accuracy=}")
        # accelerator.print(f"{eval_preds[:10]=}")
        # accelerator.print(f"{dataset['train'][label_column][:10]=}")

    # if do_test:
    #     model.eval()
    #     test_preds = []
    #     for _, batch in enumerate(tqdm(test_dataloader)):
    #         batch = {k: v for k, v in batch.items() if k != "labels"}
    #         with torch.no_grad():
    #             outputs = accelerator.unwrap_model(model).generate(
    #                 **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
    #             )  # synced_gpus=True for DS-stage 3
    #         outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
    #         preds = accelerator.gather(outputs)
    #         preds = preds[:, max_length:].detach().cpu().numpy()
    #         test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

    #     test_preds_cleaned = []
    #     for _, pred in enumerate(test_preds):
    #         test_preds_cleaned.append(get_closest_label(pred, classes))

    #     test_df = dataset["test"].to_pandas()
    #     assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
    #     test_df[label_column] = test_preds_cleaned
    #     test_df["text_labels_orig"] = test_preds
    #     accelerator.print(test_df[[text_column, label_column]].sample(20))

    #     pred_df = test_df[["ID", label_column]]
    #     pred_df.columns = ["ID", "Label"]

    #     os.makedirs(f"data/{dataset_name}", exist_ok=True)
    #     pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)

    accelerator.wait_for_everyone()
    # Option1: Pushing the model to Hugging Face Hub
    # model.push_to_hub(
    #     f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
    #     token = "hf_..."
    # )
    # token (`bool` or `str`, *optional*):
    #     `token` is to be used for HTTP Bearer authorization when accessing remote files. If `True`, will use the token generated
    #     when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
    #     is not specified.
    #     Or you can get your token from https://huggingface.co/settings/token
    # Option2: Saving the model locally
    peft_model_id = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace(
        "/", "_"
    )
    model.save_pretrained(peft_model_id)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()