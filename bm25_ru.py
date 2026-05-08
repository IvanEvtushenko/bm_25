from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

from razdel import tokenize as razdel_tokenize
import pymorphy3

PUNCT = set(string.punctuation + "«»—–…„""''")
TOKEN_RE = re.compile(r"^[\wа-яА-ЯёЁ\-]+$", re.UNICODE)

RU_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "это", "также", "которые", "который", "которая", "которых", "которой",
}


@dataclass
class Tokenizer:
    morph: pymorphy3.MorphAnalyzer = field(default_factory=pymorphy3.MorphAnalyzer)
    stopwords: set[str] = field(default_factory=lambda: set(RU_STOPWORDS))
    min_len: int = 2
    _cache: dict[str, str] = field(default_factory=dict)

    def _lemma(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        parses = self.morph.parse(token)
        lemma = parses[0].normal_form if parses else token
        self._cache[token] = lemma
        return lemma

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        for t in razdel_tokenize(text or ""):
            raw = t.text.lower()
            if not raw or raw in PUNCT:
                continue
            if not TOKEN_RE.match(raw):
                continue
            if len(raw) < self.min_len:
                continue
            lemma = self._lemma(raw)
            if lemma in self.stopwords or len(lemma) < self.min_len:
                continue
            out.append(lemma)
        return out
