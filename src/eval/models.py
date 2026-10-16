"""Model ve tokenizer yükleme (configs/models.yaml + configs/arms.yaml).

Desteklenen kollar: `fp16` (referans, `reference_dtype`) ve bitsandbytes kolları (`bnb_int8`,
`bnb_nf4`). bitsandbytes kollarında nicemlenmeyen katmanlar (gömme, normalizasyon, lm_head)
referans veri tipinde kalır (DECISIONS.md D-014).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def bnb_config(arm_cfg: dict) -> BitsAndBytesConfig:
    """`configs/arms.yaml` içindeki bir bitsandbytes kolundan nicemleme yapılandırması."""
    if arm_cfg["bits"] == 8:
        return BitsAndBytesConfig(load_in_8bit=True,
                                  llm_int8_threshold=arm_cfg["llm_int8_threshold"])
    if arm_cfg["bits"] == 4:
        return BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type=arm_cfg["quant_type"],
                                  bnb_4bit_use_double_quant=arm_cfg["double_quant"],
                                  bnb_4bit_compute_dtype=DTYPES[arm_cfg["compute_dtype"]])
    raise ValueError(f"Desteklenmeyen bit sayısı: {arm_cfg['bits']}")


def load(model_cfg: dict, arm: str, arms_cfg: dict, device: str = "cuda"):
    """Bir modeli verilen kolda yükler."""
    arm_cfg = arms_cfg["arms"][arm]
    dtype = DTYPES[arms_cfg["reference_dtype"]]
    kwargs = {"revision": model_cfg["revision"]}
    if not model_cfg.get("gated"):
        kwargs["token"] = False  # açık modellerde kayıtlı (olası geçersiz) token'ı gönderme
    tok = AutoTokenizer.from_pretrained(model_cfg["hf_id"], **kwargs)

    if arm_cfg["method"] == "none":
        model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_id"], dtype=dtype, **kwargs)
        model.to(device)
    elif arm_cfg["method"] == "bitsandbytes":
        # Nicemlenmiş ağırlıklar yükleme sırasında GPU'ya yerleşir; sonradan .to() yapılamaz.
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["hf_id"], dtype=dtype, quantization_config=bnb_config(arm_cfg),
            device_map={"": device}, **kwargs)
    else:
        raise NotImplementedError(f"'{arm}' kolu ({arm_cfg['method']}) yerelde uygulanmadı")
    model.eval()
    return model, tok


def quantization_summary(model) -> dict:
    """Nicemlenmiş ve nicemlenmemiş doğrusal katman sayıları (kaydı doğrulamak için)."""
    import bitsandbytes as bnb

    counts = {"linear8bit": 0, "linear4bit": 0, "linear_unquantized": []}
    for name, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear8bitLt):
            counts["linear8bit"] += 1
        elif isinstance(mod, bnb.nn.Linear4bit):
            counts["linear4bit"] += 1
        elif isinstance(mod, torch.nn.Linear):
            counts["linear_unquantized"].append(name)
    return counts
