"""Cümle bazlı NLL (PROTOCOL §7.3).

Her cümle ayrı bir dizi olarak değerlendirilir: [önek] + cümle token'ları. Önek, modelin BOS
token'ıdır; BOS'u olmayan modellerde (ör. Qwen2.5) EOS token'ı aynı rolde kullanılır
(DECISIONS.md D-007). Cümlenin TÜM token'ları skorlanır; önek skorlanmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SentenceScore:
    nll_nat: float
    n_tokens: int


def prefix_token_id(tokenizer) -> int:
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer'da BOS veya EOS token'ı yok")


def encode(tokenizer, texts: list[str], add_prefix: bool = True) -> list[list[int]]:
    prefix = [prefix_token_id(tokenizer)] if add_prefix else []
    return [prefix + tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


@torch.no_grad()
def score_ids(model, ids_batch: list[list[int]], pad_id: int) -> list[SentenceScore]:
    """Sağdan dolgulu tek bir batch için cümle başına toplam NLL (nat)."""
    device = next(model.parameters()).device
    max_len = max(len(x) for x in ids_batch)
    input_ids = torch.full((len(ids_batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids_batch), max_len), dtype=torch.long)
    for i, ids in enumerate(ids_batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1]
    targets = input_ids[:, 1:]
    # Tam sözlük üzerinde float32 log-softmax'ı satır satır al: 4 GB VRAM'de tüm batch'i
    # birden float32'ye çevirmek belleği aşıyor.
    token_lp = torch.stack([
        F.log_softmax(logits[i].float(), dim=-1).gather(-1, targets[i].unsqueeze(-1)).squeeze(-1)
        for i in range(logits.shape[0])
    ])
    del logits

    # Hedef konumlar 1..L-1: önek (konum 0) skorlanmaz, dolgu maskelenir.
    mask = attn[:, 1:]
    nll = -(token_lp * mask).sum(dim=-1)
    n_tok = mask.sum(dim=-1)
    return [SentenceScore(float(a), int(b)) for a, b in zip(nll.cpu(), n_tok.cpu())]


def score_texts(model, tokenizer, texts: list[str], batch_size: int = 8,
                add_prefix: bool = True) -> list[SentenceScore]:
    ids = encode(tokenizer, texts, add_prefix=add_prefix)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else prefix_token_id(tokenizer)
    out: list[SentenceScore] = []
    for start in range(0, len(ids), batch_size):
        out.extend(score_ids(model, ids[start : start + batch_size], pad_id))
    return out
