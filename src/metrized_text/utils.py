from collections import OrderedDict


def copy_state_dict(state_dict):
    start_idx = 1 if list(state_dict.keys())[0].startswith("module") else 0
    return OrderedDict(
        (".".join(k.split(".")[start_idx:]), v) for k, v in state_dict.items()
    )


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
