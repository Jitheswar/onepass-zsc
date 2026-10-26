"""User-facing API.

    from onepass import ZeroShotClassifier
    clf = ZeroShotClassifier.load("checkpoints/onepass-xs")
    clf("I loved this movie!", ["positive", "negative"])
    # {'positive': 0.97, 'negative': 0.03}

Single-label mode (default): probabilities over the given labels sum to 1.
Multi-label mode: each label gets its own independent probability.
Both are calibrated with constants stored in calibration.json.
"""
import json
import os

import torch

from .model import Encoder, OnePassModel, build


class ZeroShotClassifier:
    def __init__(self, model, tok, enc, calib=None, device="cpu", label_chunk=24):
        self.model = model.eval().to(device)
        self.tok = tok
        self.enc = enc
        self.device = device
        self.label_chunk = label_chunk
        self.calib = calib or {"softmax_T": 1.0, "sigmoid_a": 1.0, "sigmoid_b": 0.0}

    @classmethod
    def load(cls, path, device=None, max_len=512, dtype=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.load(open(os.path.join(path, "config.json")))
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        enc_dir = os.path.join(path, "encoder")
        conf = AutoConfig.from_pretrained(enc_dir if os.path.isdir(enc_dir) else cfg["backbone"])
        model = OnePassModel(cfg["backbone"], nli_head=cfg.get("nli_head", False), config=conf)
        model.encoder.resize_token_embeddings(max(len(tok), model.encoder.config.vocab_size))
        st = os.path.join(path, "model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file
            sd = load_file(st)
        else:
            sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        model.load_state_dict(sd)
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        model = model.to(dtype)
        cp = os.path.join(path, "calibration.json")
        calib = json.load(open(cp)) if os.path.exists(cp) else None
        return cls(model, tok, Encoder(tok, max_len=max_len, layout=cfg.get("layout", "labels_first"), template=cfg.get("template")), calib, device)

    def _chunks(self, labels):
        n = self.label_chunk
        return [labels[i: i + n] for i in range(0, len(labels), n)]

    @torch.inference_mode()
    def logits(self, texts, labels):
        """Raw scores, shape (len(texts), len(labels)). One forward pass per batch:
        if there are more labels than `label_chunk`, each text is repeated once per
        label chunk inside the same batch."""
        chunks = self._chunks(list(labels))
        T, L = [], []
        for t in texts:
            for c in chunks:
                T.append(t)
                L.append(c)
        b = self.enc(T, L)
        b = {k: v.to(self.device) for k, v in b.items()}
        out = self.model(**b).float()
        rows = []
        k = len(chunks)
        for i in range(len(texts)):
            parts = [out[i * k + j, : len(chunks[j])] for j in range(k)]
            rows.append(torch.cat(parts))
        return torch.stack(rows).cpu()

    def probs_from_logits(self, logits, multi_label=False):
        c = self.calib
        if multi_label:
            return torch.sigmoid(c["sigmoid_a"] * logits + c["sigmoid_b"])
        return torch.softmax(logits / c["softmax_T"], dim=-1)

    def predict(self, texts, labels, multi_label=False, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            lg = self.logits(texts[i: i + batch_size], labels)
            out.append(self.probs_from_logits(lg, multi_label))
        return torch.cat(out)

    def ask(self, text, questions, multi_label=False):
        """Answer several label questions about one text in a single forward pass.

        questions: {"sentiment": ["positive", "negative"], "topic": ["sports", "politics", "tech"]}
        Returns {"sentiment": {...probs...}, "topic": {...probs...}}; probabilities are
        normalised within each question (or independent per label if multi_label=True).
        """
        flat, spans = [], {}
        for q, labels in questions.items():
            spans[q] = (len(flat), len(flat) + len(labels))
            flat.extend(labels)
        lg = self.logits([text], flat)[0]
        out = {}
        for q, (a, b) in spans.items():
            p = self.probs_from_logits(lg[a:b].unsqueeze(0), multi_label)[0]
            out[q] = dict(sorted(zip(questions[q], p.tolist()), key=lambda kv: -kv[1]))
        return out

    def __call__(self, text, labels, multi_label=False):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        p = self.predict(texts, list(labels), multi_label)
        res = [dict(sorted(zip(labels, row.tolist()), key=lambda kv: -kv[1])) for row in p]
        return res[0] if single else res
