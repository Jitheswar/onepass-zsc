"""One-pass zero-shot classifier.

The text and every candidate label go through the encoder together in one sequence:

    [CLS] [L] label_1 [L] label_2 ... [L] label_k [T] text ... [SEP]

Each label gets a vector (mean of its marker + tokens), the text gets a vector
(mean of its marker + tokens), and a small head scores every (label, text) pair.
Because all labels share one forward pass, cost barely grows with the number of labels.
"""
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

LABEL_TOKEN = "[L]"
TEXT_TOKEN = "[T]"


def masked_mean(h, mask):
    # h: (B, S, D), mask: (B, N, S) -> (B, N, D)
    w = mask.to(h.dtype)
    return torch.bmm(w, h) / w.sum(-1, keepdim=True).clamp(min=1.0)


class OnePassModel(nn.Module):
    def __init__(self, backbone_name, hidden_head=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder = AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)
        d = self.encoder.config.hidden_size
        h = hidden_head or d
        self.head = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask, label_mask, text_mask, label_valid):
        """label_mask: (B, N, S) token membership of each label span.
        text_mask: (B, S) token membership of the text span.
        label_valid: (B, N) bool, False for padding labels.
        Returns raw logits (B, N); padded labels get -inf-ish values."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        lab = masked_mean(hs, label_mask)  # (B, N, D)
        txt = masked_mean(hs, text_mask.unsqueeze(1)).expand_as(lab)  # (B, N, D)
        feats = torch.cat([lab, txt, lab * txt, (lab - txt).abs()], dim=-1)
        logits = self.head(feats).squeeze(-1)
        return logits.masked_fill(~label_valid, -1e4)


class Encoder:
    """Turns (text, labels) pairs into model inputs."""

    def __init__(self, tokenizer, max_len=256, max_label_tokens=16):
        self.tok = tokenizer
        self.max_len = max_len
        self.max_label_tokens = max_label_tokens
        self.lid = tokenizer.convert_tokens_to_ids(LABEL_TOKEN)
        self.tid = tokenizer.convert_tokens_to_ids(TEXT_TOKEN)

    def __call__(self, texts, label_lists):
        tok = self.tok
        B = len(texts)
        N = max(len(l) for l in label_lists)
        all_labels = sorted({l for ls in label_lists for l in ls})
        lab_ids = dict(zip(all_labels, tok(all_labels, add_special_tokens=False)["input_ids"]))
        txt_ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        seqs, lspans, tspans = [], [], []
        for t_ids, labels in zip(txt_ids, label_lists):
            ids = [tok.cls_token_id]
            spans = []
            for l in labels:
                s = len(ids)
                ids.append(self.lid)
                ids.extend(lab_ids[l][: self.max_label_tokens])
                spans.append((s, len(ids)))
            budget = self.max_len - len(ids) - 2
            if budget < 8:
                raise ValueError("Too many / too long labels for max_len=%d" % self.max_len)
            s = len(ids)
            ids.append(self.tid)
            ids.extend(t_ids[:budget])
            tspans.append((s, len(ids)))
            ids.append(tok.sep_token_id)
            seqs.append(ids)
            lspans.append(spans)
        S = max(len(s) for s in seqs)
        input_ids = torch.full((B, S), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((B, S), dtype=torch.long)
        label_mask = torch.zeros((B, N, S), dtype=torch.bool)
        text_mask = torch.zeros((B, S), dtype=torch.bool)
        label_valid = torch.zeros((B, N), dtype=torch.bool)
        for i, (ids, spans, (ts, te)) in enumerate(zip(seqs, lspans, tspans)):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
            text_mask[i, ts:te] = True
            for j, (a, b) in enumerate(spans):
                label_mask[i, j, a:b] = True
                label_valid[i, j] = True
        return dict(input_ids=input_ids, attention_mask=attn, label_mask=label_mask,
                    text_mask=text_mask, label_valid=label_valid)


def build(backbone_name, max_len=256):
    tok = AutoTokenizer.from_pretrained(backbone_name)
    tok.add_special_tokens({"additional_special_tokens": [LABEL_TOKEN, TEXT_TOKEN]})
    model = OnePassModel(backbone_name)
    if len(tok) > model.encoder.get_input_embeddings().weight.shape[0]:
        model.encoder.resize_token_embeddings(len(tok))
    with torch.no_grad():
        w = model.encoder.get_input_embeddings().weight
        base = w[[tok.cls_token_id, tok.sep_token_id]].mean(0)
        for t in (LABEL_TOKEN, TEXT_TOKEN):
            w[tok.convert_tokens_to_ids(t)] = base + 0.02 * torch.randn_like(base)
    return model, tok, Encoder(tok, max_len=max_len)
