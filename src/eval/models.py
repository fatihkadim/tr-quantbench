"""Model ve tokenizer yükleme (configs/models.yaml + configs/arms.yaml)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load(model_cfg: dict, arm: str, arms_cfg: dict, device: str = "cuda"):
    """Bir modeli verilen kolda yükler. Faz 1'de yalnızca `fp16` (referans) kolu desteklenir."""
    if arm != "fp16":
        raise NotImplementedError(f"'{arm}' kolu Faz 3'te eklenecek")
    dtype = DTYPES[arms_cfg["reference_dtype"]]
    kwargs = {"revision": model_cfg["revision"]}
    if not model_cfg.get("gated"):
        kwargs["token"] = False  # açık modellerde kayıtlı (olası geçersiz) token'ı gönderme
    tok = AutoTokenizer.from_pretrained(model_cfg["hf_id"], **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], dtype=dtype, **kwargs)
    model.to(device).eval()
    return model, tok
