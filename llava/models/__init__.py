import torch


MODEL_FACTORY = {

}


def get_model(model_args, training_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    cfg_pretrained = None

    if model_args.vision_tower is not None:
