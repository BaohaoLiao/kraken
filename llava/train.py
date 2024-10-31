import torch
import transformers
from transformers.utils import check_min_version

from llava.args import parse_args
from llava.utils import (
    rank0_print,
)

check_min_version("4.43.0.dev0")
local_rank = None


def train(attn_implementation=None):
    global local_rank

    model_args, data_args, training_args = parse_args()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)



if __name__ == "__main__":
    train()