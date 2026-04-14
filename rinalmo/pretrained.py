import json
import re
from pathlib import Path

import gdown
import torch

from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo
from rinalmo.config import model_config
from rinalmo.inference_device import select_inference_device

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rinalmo_pretrained"

with open(Path(__file__).parent / "resources" / "model2gdisk.json", "r") as f:
    MODEL_TO_GDISK_ID = json.load(f)
AVAILABLE_MODELS = MODEL_TO_GDISK_ID.keys()

_WQKV_KEY = re.compile(r"^transformer\.blocks\.(\d+)\.mh_attn\.Wqkv\.(weight|bias)$")
_OUT_PROJ_KEY = re.compile(r"^transformer\.blocks\.(\d+)\.mh_attn\.out_proj\.(weight|bias)$")


def _adapt_flat_mh_attn_checkpoint(state_dict: dict) -> dict:
    """
    Older checkpoints use fused QKV (Wqkv) and out_proj on MultiHeadSelfAttention;
    current code uses nested MultiHeadAttention (mh_attn.mh_attn) with to_q/to_k/to_v.
    """
    if not any(_WQKV_KEY.match(k) for k in state_dict):
        return state_dict

    adapted: dict = {}
    for key, tensor in state_dict.items():
        m = _WQKV_KEY.match(key)
        if m:
            block_idx, param = m.groups()
            base = f"transformer.blocks.{block_idx}.mh_attn.mh_attn"
            q, k_, v_ = tensor.chunk(3, dim=0)
            adapted[f"{base}.to_q.{param}"] = q
            adapted[f"{base}.to_k.{param}"] = k_
            adapted[f"{base}.to_v.{param}"] = v_
            continue
        m = _OUT_PROJ_KEY.match(key)
        if m:
            block_idx, param = m.groups()
            adapted[f"transformer.blocks.{block_idx}.mh_attn.mh_attn.out_proj.{param}"] = tensor
            continue
        adapted[key] = tensor
    return adapted


def download_pretrained_model(model_name: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=MODEL_TO_GDISK_ID[model_name], output=str(
        local_path.resolve()))


def get_pretrained_model(
    model_name: str,
    force_download: bool = False,
    lm_config: str = "giga",
    device: torch.device | str | None = None,
):
    assert model_name in AVAILABLE_MODELS, f"Model '{model_name}' is not available! Available models: {AVAILABLE_MODELS}"
    pretrained_weights_path = DEFAULT_CACHE_DIR / f"{model_name}.pt"

    if force_download or not pretrained_weights_path.exists():
        download_pretrained_model(model_name, pretrained_weights_path)

    if device is None:
        device = select_inference_device()
    else:
        device = torch.device(device)

    config = model_config(lm_config)
    if device.type != "cuda":
        config.model.transformer.use_flash_attn = False
    else:
        try:
            import rinalmo.model.attention_flash  # noqa: F401
        except ImportError:
            config.model.transformer.use_flash_attn = False

    model = RiNALMo(config)
    alphabet = Alphabet(**config['alphabet'])
    raw = torch.load(pretrained_weights_path, map_location=device)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    state_dict = _adapt_flat_mh_attn_checkpoint(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_suffix = ".rotary_emb.inv_freq"
    bad_missing = [k for k in incompatible.missing_keys if not k.endswith(allowed_missing_suffix)]
    if bad_missing:
        raise RuntimeError(
            "Checkpoint is missing parameters expected by the model (showing up to 10): "
            f"{bad_missing[:10]}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint has unexpected keys (showing up to 10): "
            f"{incompatible.unexpected_keys[:10]}"
        )
    model = model.to(device)

    return model, alphabet
