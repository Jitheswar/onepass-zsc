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
    def __init__(self, backbone_name, hidden_head=None, nli_head=False, config=None):
        """nli_head=True: the backbone is an NLI cross-encoder; its entailment classifier is reused,
        applied at each label's [L] marker, as the starting score (the MLP head starts at zero)."""
        super().__init__()
        self.backbone_name = backbone_name
        self.nli_head = nli_head
        if nli_head:
            from transformers import AutoModelForSequenceClassification
            if config is not None:
                full = AutoModelForSequenceClassification.from_config(config)
            else:
                full = AutoModelForSequenceClassification.from_pretrained(backbone_name, torch_dtype=torch.float32)
            self.encoder = getattr(full, full.base_model_prefix)
            self.pool_dense = full.pooler.dense
            self.cls = full.classifier
            id2 = {v.lower(): k for k, v in full.config.id2label.items()}
            self.ent = id2["entailment"]
            self.non = [i for i in range(full.config.num_labels) if i != self.ent]
        else:
            self.encoder = AutoModel.from_config(config) if config is not None else \
                AutoModel.from_pretrained(backbone_name, torch_dtype=torch.float32)
        d = self.encoder.config.hidden_size
        h = hidden_head or d
        self.head = nn.Sequential(nn.Linear(4 * d, h), nn.GELU(), nn.Linear(h, 1))
        if nli_head:
            nn.init.zeros_(self.head[2].weight)
            nn.init.zeros_(self.head[2].bias)

    def forward(self, input_ids, attention_mask, label_mask, text_mask, label_valid, rel_pos=None, valid=None):
        """label_mask: (B, N, S) token membership of each label span.
        text_mask: (B, S) token membership of the text span.
        label_valid: (B, N) bool, False for padding labels.
        rel_pos/valid: only for the packed layout (3-D attention mask + shared virtual positions).
        Returns raw logits (B, N); padded labels get -inf-ish values."""
        if rel_pos is not None:
            emb = self.encoder.embeddings(input_ids=input_ids, mask=valid.long())
            hs = self.encoder.encoder(emb, attention_mask, output_hidden_states=False,
                                      relative_pos=rel_pos.squeeze(0) if rel_pos.dim() == 4 else rel_pos[:1]).last_hidden_state
        else:
            hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lab = masked_mean(hs, label_mask)  # (B, N, D)
        txt = masked_mean(hs, text_mask.unsqueeze(1)).expand_as(lab)  # (B, N, D)
        feats = torch.cat([lab, txt, lab * txt, (lab - txt).abs()], dim=-1)
        logits = self.head(feats).squeeze(-1)
        if self.nli_head:
            first = label_mask.float().argmax(-1)  # position of each label's [L] marker
            hm = torch.gather(hs, 1, first.unsqueeze(-1).expand(-1, -1, hs.shape[-1]))
            z = self.cls(torch.nn.functional.gelu(self.pool_dense(hm)))
            logits = logits + z[..., self.ent] - torch.logsumexp(z[..., self.non], -1)
        return logits.masked_fill(~label_valid, -1e4)


class Encoder:
    """Turns (text, labels) pairs into model inputs."""

    def __init__(self, tokenizer, max_len=256, max_label_tokens=16, layout="labels_first", template=None):
        """layout: "labels_first" = [CLS] [L] l1 [L] l2 ... [T] text [SEP]
                   "text_first"   = [CLS] [T] text [SEP] [L] l1 [L] l2 ... [SEP]  (mirrors NLI premise/hypothesis order)"""
        assert layout in ("labels_first", "text_first")
        self.layout = layout
        self.template = template  # e.g. "This text is about {}." wraps every label as an NLI-style hypothesis
        if template:
            self.max_label_tokens = max_label_tokens + 12
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
        shown = [self.template.format(l) if self.template else l for l in all_labels]
        lab_ids = dict(zip(all_labels, tok(shown, add_special_tokens=False)["input_ids"]))
        txt_ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_len)["input_ids"]
        seqs, lspans, tspans = [], [], []
        for t_ids, labels in zip(txt_ids, label_lists):
            if self.layout == "text_first":
                lab_len = sum(1 + len(lab_ids[l][: self.max_label_tokens]) for l in labels)
                budget = self.max_len - lab_len - 4
                if budget < 8:
                    raise ValueError("Too many / too long labels for max_len=%d" % self.max_len)
                ids = [tok.cls_token_id, self.tid] + t_ids[:budget]
                tspans.append((1, len(ids)))
                ids.append(tok.sep_token_id)
                spans = []
                for l in labels:
                    s = len(ids)
                    ids.append(self.lid)
                    ids.extend(lab_ids[l][: self.max_label_tokens])
                    spans.append((s, len(ids)))
                ids.append(tok.sep_token_id)
                seqs.append(ids)
                lspans.append(spans)
                continue
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


def build(backbone_name, max_len=256, nli_head=False, layout="labels_first", template=None):
    if layout == "packed":
        tok = AutoTokenizer.from_pretrained(backbone_name)
        model = OnePassModel(backbone_name, nli_head=True)
        return model, tok, PackedEncoder(tok, max_len=max_len, template=template or "This text is about {}.")
    tok = AutoTokenizer.from_pretrained(backbone_name)
    tok.add_special_tokens({"additional_special_tokens": [LABEL_TOKEN, TEXT_TOKEN]})
    model = OnePassModel(backbone_name, nli_head=nli_head)
    if len(tok) > model.encoder.get_input_embeddings().weight.shape[0]:
        model.encoder.resize_token_embeddings(len(tok))
    with torch.no_grad():
        w = model.encoder.get_input_embeddings().weight
        base = w[[tok.cls_token_id, tok.sep_token_id]].mean(0)
        for t in (LABEL_TOKEN, TEXT_TOKEN):
            w[tok.convert_tokens_to_ids(t)] = base + 0.02 * torch.randn_like(base)
    return model, tok, Encoder(tok, max_len=max_len, layout=layout, template=template)


# ---------------------------------------------------------------------------------------------
# "Packed NLI" one-pass layout:
#   [CLS] text [SEP] [pad..] | [CLS] hyp_1 [SEP] [pad..] | [CLS] hyp_2 [SEP] [pad..] | ...
# Text tokens attend only to the text; each hypothesis block attends to the text and to itself
# (never to other labels), and every block gets the same virtual positions right after the text.
# So each label is scored like an ordinary NLI (text, hypothesis) pair, but the text is encoded once
# and all labels share one forward pass. The backbone's own entailment head scores each block.

def hypothesis(label, template="This text is about {}."):
    """Short labels become NLI hypotheses; labels that already read like sentences are kept."""
    l = label.strip()
    if l.endswith(".") or len(l.split()) > 8:
        return l
    return template.format(l)


class PackedEncoder:
    def __init__(self, tokenizer, max_len=512, max_text=256, max_hyp=24, template="This text is about {}.",
                 bucket_size=256, max_position=512):
        self.tok, self.max_len, self.max_text, self.max_hyp, self.template = tokenizer, max_len, max_text, max_hyp, template
        self.bucket_size, self.max_position = bucket_size, max_position
        self.layout = "packed"

    def __call__(self, texts, label_lists):
        from transformers.models.deberta_v2.modeling_deberta_v2 import make_log_bucket_position
        tok = self.tok
        B, N = len(texts), max(len(l) for l in label_lists)
        all_labels = sorted({l for ls in label_lists for l in ls})
        hyp_ids = dict(zip(all_labels, tok([hypothesis(l, self.template) for l in all_labels],
                                           add_special_tokens=False)["input_ids"]))
        Lh = max(len(hyp_ids[l][: self.max_hyp]) for l in all_labels) + 2  # [CLS] hyp [SEP]
        txt = tok(list(texts), add_special_tokens=False, truncation=True, max_length=self.max_text)["input_ids"]
        tmax = max(len(t) for t in txt)
        tmax = max(1, min(tmax, self.max_len - N * Lh - 2))
        R = tmax + 2
        S = R + N * Lh
        ids = torch.full((B, S), tok.pad_token_id, dtype=torch.long)
        valid = torch.zeros((B, S), dtype=torch.bool)
        text_tok = torch.zeros((B, S), dtype=torch.bool)
        block = torch.full((B, S), -1, dtype=torch.long)  # -1 = text region, j = label slot j
        label_mask = torch.zeros((B, N, S), dtype=torch.bool)
        label_valid = torch.zeros((B, N), dtype=torch.bool)
        for i, (t, labels) in enumerate(zip(txt, label_lists)):
            t = t[:tmax]
            seq = [tok.cls_token_id] + t + [tok.sep_token_id]
            ids[i, : len(seq)] = torch.tensor(seq)
            valid[i, : len(seq)] = True
            text_tok[i, 1: 1 + len(t)] = True
            for j, l in enumerate(labels):
                h = [tok.cls_token_id] + hyp_ids[l][: self.max_hyp] + [tok.sep_token_id]
                a = R + j * Lh
                ids[i, a: a + len(h)] = torch.tensor(h)
                valid[i, a: a + len(h)] = True
                block[i, a: a + Lh] = j
                label_mask[i, j, a: a + len(h)] = True
                label_valid[i, j] = True
        same = block.unsqueeze(2) == block.unsqueeze(1)
        to_text = (block == -1).unsqueeze(1).expand(B, S, S)
        attn = (same | to_text) & valid.unsqueeze(1) & valid.unsqueeze(2)
        # virtual positions: text region 0..R-1, every label slot restarts at R
        vp = torch.arange(S)
        vp = torch.where(vp >= R, R + (vp - R) % Lh, vp)
        rel = vp[:, None] - vp[None, :]
        rel = make_log_bucket_position(rel, self.bucket_size, self.max_position).long().unsqueeze(0)
        return dict(input_ids=ids, attention_mask=attn.long(), label_mask=label_mask, text_mask=text_tok,
                    label_valid=label_valid, rel_pos=rel, valid=valid)
