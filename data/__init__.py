from .samsum_dataset import get_preprocessed_samsum
from .summscreen_dataset import get_preprocessed_summscreen
from .mediasum_dataset import get_preprocessed_mediasum
from .friendsqa_dataset import get_preprocessed_friendsqa
from .data_collator import DataCollatorForSeq2Seq
from .metrics import get_summarization_metrics, get_qa_metrics

def get_dataset_and_metrics(name, tokenizer, return_token_type_ids=False, test_only=False):
    if name == 'samsum':
        dataset = get_preprocessed_samsum(tokenizer, return_token_type_ids=return_token_type_ids, test_only=test_only)
        compute_metrics = get_summarization_metrics(tokenizer)
    elif name == 'summscreen':
        dataset = get_preprocessed_summscreen(tokenizer, return_token_type_ids=return_token_type_ids, test_only=test_only)
        compute_metrics = get_summarization_metrics(tokenizer)
    elif name == 'mediasum':
        dataset = get_preprocessed_mediasum(tokenizer, return_token_type_ids=return_token_type_ids, test_only=test_only)
        compute_metrics = get_summarization_metrics(tokenizer)
    elif name == 'friendsqa':
        dataset = get_preprocessed_friendsqa(tokenizer, return_token_type_ids=return_token_type_ids, test_only=test_only)
        compute_metrics = get_qa_metrics(tokenizer)
    else:
        raise NotImplementedError(f"Dataset {name} not supported.")
    return dataset, compute_metrics
