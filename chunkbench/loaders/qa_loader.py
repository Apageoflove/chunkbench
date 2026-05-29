from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from .document_loader import Document


class QALoader:
    def load(self, path: str) -> List[dict]:
        ext = Path(path).suffix.lower()
        if ext == ".jsonl":
            return self._load_jsonl(path)
        if ext == ".json":
            return self._load_json(path)
        if ext == ".csv":
            return self._load_csv(path)
        raise ValueError(f"Unsupported QA format: {ext}")

    def validate(self, qa_pairs: List[dict]) -> bool:
        for pair in qa_pairs:
            if "question" not in pair or "answer" not in pair:
                return False
        return True

    def generate_simple_qa(self, documents: List[Document], n: int = 20) -> List[dict]:
        qa_pairs = []
        for doc in documents:
            sentences = self._split_sentences(doc.text)
            sampled = random.sample(sentences, min(n // len(documents), len(sentences)))
            for sent in sampled:
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                qa_pairs.append({
                    "question": self._make_question(sent),
                    "answer": sent,
                    "doc_id": doc.doc_id,
                })
                if len(qa_pairs) >= n:
                    return qa_pairs
        return qa_pairs

    # -- private helpers --

    @staticmethod
    def _load_jsonl(path: str) -> List[dict]:
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs

    @staticmethod
    def _load_json(path: str) -> List[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    @staticmethod
    def _load_csv(path: str) -> List[dict]:
        pairs = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(dict(row))
        return pairs

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[。！？.!?])\s*', text)
        return [p.strip() for p in parts if len(p.strip()) > 10]

    @staticmethod
    def _make_question(answer: str) -> str:
        if len(answer) > 40:
            return f"以下内容描述了什么：{answer[:40]}……？"
        return f"请解释以下内容：{answer}"
