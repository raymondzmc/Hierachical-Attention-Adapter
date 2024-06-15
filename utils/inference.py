import torch
from transformers import StoppingCriteria


class StopSequenceRepeatCriteria(StoppingCriteria):

    def __init__(self, tokenizer, stop_sequences=[], max_repeat=None):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.max_repeat = max_repeat

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # not compatible with batch size > 1
        if input_ids.shape[0] > 1:
            return False
        
        input_ids = input_ids[0]
        
        # Check if last `max_repeat` tokens are the same
        if torch.unique(input_ids[-self.max_repeat:]).numel() == 1:
            return True

        # Check if generation ends with any of the `stop_sequences ``
        decoded_string = self.tokenizer.decode(input_ids)
        for stop in self.stop_sequences:
            if decoded_string.endswith(stop):
                return True
        return False