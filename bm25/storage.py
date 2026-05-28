"""Persistence: BM25-индекс ↔ файлы в data/index/, документы ↔ data/docs.jsonl.

Источник правды — `docs.jsonl` (одна запись на строку, append-only).
Индекс — производный numpy/scipy-кэш, всегда пересобирается из docs.jsonl.

LazyDocuments — ленивый доступ к docs.jsonl: при поиске вместо загрузки
всего файла в dict держим в RAM только offsets[N] и подгружаем нужные
строки точечно. На 1.7M документов это разница 14 МБ vs 3–5 ГБ RAM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import load_npz, save_npz

from .builder import INDEX_BLACKLIST
from .index import BM25Index, DEFAULT_B, DEFAULT_K1, INDEX_VERSION


# ─── BM25-индекс на диск и обратно ──────────────────────────────────────────
# Файлы в data/index/:
#   tf.npz          разреженная term-doc матрица (CSR)
#   doc_lens.npy    длины документов в токенах
#   df.npy          document frequency на термин
#   offsets.npy     байтовые позиции каждой строки в docs.jsonl
#   vocab.json      {лемма: column_index}
#   doc_ids.json    порядок строк tf — какой doc_id на какой строке
#   kinds.json      тип каждого документа (параллельно doc_ids)
#   pub_ids.json    pub_id каждого документа (параллельно doc_ids)
#   meta.json       версия, N, vocab_size, avgdl, k1, b, fields_by_kind

def save_index(index_dir: Path, index: BM25Index, offsets: np.ndarray | None = None) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    save_npz(index_dir / "tf.npz", index.tf)
    np.save(index_dir / "doc_lens.npy", index.doc_lens)
    np.save(index_dir / "df.npy", index.df)
    if offsets is not None:
        np.save(index_dir / "offsets.npy", offsets.astype(np.int64))
    (index_dir / "vocab.json").write_text(
        json.dumps(index.vocab, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "doc_ids.json").write_text(
        json.dumps(index.doc_ids, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "kinds.json").write_text(
        json.dumps(index.kinds, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "pub_ids.json").write_text(
        json.dumps(index.pub_ids, ensure_ascii=False), encoding="utf-8"
    )
    meta = {
        "version": INDEX_VERSION,
        "n_docs": index.n_docs,
        "vocab_size": index.vocab_size,
        "avgdl": index.avgdl,
        "k1": index.k1,
        "b": index.b,
        "index_blacklist": sorted(INDEX_BLACKLIST),
        "kind_counts": {k: index.kinds.count(k) for k in set(index.kinds)},
    }
    (index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> BM25Index:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    version = int(meta.get("version", 0))
    if version < INDEX_VERSION:
        raise RuntimeError(
            f"index format version {version} is outdated (current is {INDEX_VERSION}); "
            f"remove {index_dir} and rebuild via add_doc.py"
        )
    tf = load_npz(index_dir / "tf.npz").tocsr()
    doc_lens = np.load(index_dir / "doc_lens.npy")
    df = np.load(index_dir / "df.npy")
    vocab = json.loads((index_dir / "vocab.json").read_text(encoding="utf-8"))
    doc_ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
    kinds = json.loads((index_dir / "kinds.json").read_text(encoding="utf-8"))
    pub_ids = json.loads((index_dir / "pub_ids.json").read_text(encoding="utf-8"))
    return BM25Index(
        vocab=vocab,
        doc_ids=doc_ids,
        kinds=kinds,
        pub_ids=pub_ids,
        tf=tf,
        doc_lens=doc_lens,
        df=df,
        k1=float(meta.get("k1", DEFAULT_K1)),
        b=float(meta.get("b", DEFAULT_B)),
    )


def load_offsets(index_dir: Path) -> np.ndarray:
    return np.load(index_dir / "offsets.npy")


# ─── JSONL helpers (append-only хранилище документов) ───────────────────────

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_documents_map(jsonl_path: Path) -> dict[str, dict]:
    """Все документы в один dict — для дедупликации в add_doc.py.

    Используется ТОЛЬКО при инжесте (нам надо знать какие doc_id уже есть).
    Для поиска используется LazyDocuments — экономит память на больших корпусах.
    """
    return {doc["doc_id"]: doc for doc in read_jsonl(jsonl_path)}


# ─── LazyDocuments: ленивый доступ к JSONL по offset ────────────────────────

class LazyDocuments:
    """Поведение dict[doc_id, dict], но документы читаются с диска по запросу.

    В RAM держится только маппинг doc_id → row_index (~150 МБ при 1.7M
    документов) и numpy-массив offsets (~14 МБ). Сами документы остаются
    на диске в docs.jsonl, читаются точечно через seek + readline.

    После fork() в worker-процессе __getitem__ автоматически переоткрывает
    файл-дескриптор (см. _ensure_file) — поэтому LazyDocuments безопасен
    в multiprocessing.Pool.
    """
    def __init__(self, jsonl_path: Path, doc_ids: list[str], offsets: np.ndarray):
        self._jsonl_path = jsonl_path
        self._offsets = offsets
        self._id_to_row = {did: i for i, did in enumerate(doc_ids)}
        self._file = None
        # Запоминаем pid процесса, в котором создан handle, чтобы после fork
        # обнаружить смену процесса и переоткрыть файл.
        self._owner_pid: int | None = None

    def _ensure_file(self):
        import os
        if self._file is None or self._owner_pid != os.getpid():
            self._file = self._jsonl_path.open("rb")
            self._owner_pid = os.getpid()
        return self._file

    def __getitem__(self, doc_id: str) -> dict:
        row = self._id_to_row[doc_id]
        f = self._ensure_file()
        f.seek(int(self._offsets[row]))
        line = f.readline().decode("utf-8")
        return json.loads(line)

    def get(self, doc_id: str, default=None):
        try:
            return self[doc_id]
        except KeyError:
            return default

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._id_to_row

    def __iter__(self):
        return iter(self._id_to_row)

    def __len__(self) -> int:
        return len(self._id_to_row)

    def values(self):
        for did in self._id_to_row:
            yield self[did]

    def items(self):
        for did in self._id_to_row:
            yield did, self[did]
