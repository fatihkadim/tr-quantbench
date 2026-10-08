"""NLL hesabı: dolgu sonucu değiştirmemeli (PROTOCOL §7.3)."""

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.eval.bpb import bpb, corpus_bpb
from src.eval.nll import score_ids


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def test_batch_matches_single(tiny_model):
    seqs = [[1, 5, 9, 13], [1, 7, 7, 2, 40, 41, 42, 3, 8], [1, 60], [1, 11, 12, 13, 14, 15]]
    batched = score_ids(tiny_model, seqs, pad_id=0)
    for seq, b in zip(seqs, batched):
        (single,) = score_ids(tiny_model, [seq], pad_id=0)
        assert b.n_tokens == single.n_tokens == len(seq) - 1
        assert b.nll_nat == pytest.approx(single.nll_nat, abs=1e-4)


def test_pad_id_does_not_matter(tiny_model):
    seqs = [[1, 5, 9], [1, 7, 7, 2, 40, 41]]
    a = score_ids(tiny_model, seqs, pad_id=0)
    b = score_ids(tiny_model, seqs, pad_id=96)
    assert [x.nll_nat for x in a] == pytest.approx([x.nll_nat for x in b], abs=1e-5)


def test_bpb():
    assert bpb(math.log(2) * 8, 8) == pytest.approx(1.0)
    rows = [{"nll_nat": math.log(2) * 10, "n_bytes": 5}, {"nll_nat": 0.0, "n_bytes": 5}]
    assert corpus_bpb(rows) == pytest.approx(1.0)
