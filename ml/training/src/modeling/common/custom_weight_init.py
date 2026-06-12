import torch.nn as nn
from torch.nn.parameter import UninitializedParameter

def custom_weight_init(module: nn.Module) -> None:
    if getattr(module, "_skip_custom_init", True):
        return
    elif isinstance(module, nn.RMSNorm):
        if getattr(module, "weight", None) is not None:
            nn.init.constant_(module.weight, 1)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        # Case for LazyLinear
        if isinstance(getattr(module, "weight", None), UninitializedParameter):
            return
        nn.init.xavier_normal_(module.weight)
        if getattr(module, "bias", None) is not None:
            if not isinstance(module.bias, UninitializedParameter):
                nn.init.constant_(module.bias, 0)
