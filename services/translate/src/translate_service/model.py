"""Model wrapper — the one place that knows about NLLB and language codes.

The rest of the service only knows "translate these texts toward this
FLORES-200 code" — see docs/architecture/05-contracts.md, "Translate-сервис".
Detecting each text's own language and skipping ones already in the target
language both happen here, not in main.py, for the same reason
embeddings_service/model.py owns the e5 prefix logic: it is model-specific,
not protocol-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranslationResult:
    text: str
    # ISO 639-1, best-effort from language ID — None when detection itself
    # failed (empty/too-short text). Purely informational for the caller;
    # `translated` is what actually decided whether MT ran.
    source_lang: str | None
    translated: bool


class TranslationModel(Protocol):
    @property
    def name(self) -> str: ...

    def translate(self, texts: list[str], target_lang: str) -> list[TranslationResult]: ...


# NLLB's FLORES-200 codes for the languages notes-bot's own language-ID step
# (py3langid, trained on ISO 639-1) can actually name. Deliberately not
# NLLB's full ~200-language list: an unmapped language falls back to
# "translated=False, pass the original text through" (see `translate`
# below) rather than guessing a code wrong — same degrade-not-fail
# philosophy as bot-notes' own extraction pipeline (03-ingest.md,
# "Деградация").
_FLORES_BY_ISO639_1: dict[str, str] = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "he": "heb_Hebr",
    "ar": "arb_Arab",
    "uk": "ukr_Cyrl",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "fi": "fin_Latn",
    "el": "ell_Grek",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "bg": "bul_Cyrl",
    "hi": "hin_Deva",
    "fa": "pes_Arab",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ka": "kat_Geor",
    "hy": "hye_Armn",
    "az": "azj_Latn",
    "kk": "kaz_Cyrl",
}
_ISO_BY_FLORES: dict[str, str] = {v: k for k, v in _FLORES_BY_ISO639_1.items()}

# NLLB's own limit is 512 subword tokens; texts here are already
# chunk-sized (see notes_bot.domain.chunk) or a single search query, never a
# whole raw transcript, so truncation at this length loses at most the tail
# of an unusually dense chunk rather than most of it.
_MAX_INPUT_TOKENS = 512


def _detect_iso639_1(text: str) -> str | None:
    import py3langid

    stripped = text.strip()
    if not stripped:
        return None
    lang, _confidence = py3langid.classify(stripped)
    return lang


class NllbTranslationModel:
    """Loads one NLLB checkpoint and serves it for the process lifetime.

    Constructing this is expensive (seconds, once weights are on disk) —
    same reasoning as SentenceTransformerModel for why /readyz waits for it.
    transformers/torch import deferred into __init__ so tests that only need
    the Protocol (via a fake) never pay for torch.
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        self._model.eval()

    @property
    def name(self) -> str:
        return self._model_name

    def translate(self, texts: list[str], target_lang: str) -> list[TranslationResult]:
        if not texts:
            return []

        target_iso = _ISO_BY_FLORES.get(target_lang)
        results: list[TranslationResult | None] = [None] * len(texts)
        # Grouped by source FLORES code: NLLB's tokenizer takes one
        # `src_lang` per call, so mixed-language batches (a realistic case —
        # one /search page, one note's chunks) are split into per-language
        # sub-batches instead of one text at a time.
        groups: dict[str, list[int]] = {}
        for i, text in enumerate(texts):
            iso = _detect_iso639_1(text)
            flores = _FLORES_BY_ISO639_1.get(iso) if iso else None
            if flores is None or iso == target_iso:
                results[i] = TranslationResult(text=text, source_lang=iso, translated=False)
            else:
                groups.setdefault(flores, []).append(i)

        for source_flores, indices in groups.items():
            translated = self._translate_batch(
                [texts[i] for i in indices], source_flores, target_lang
            )
            source_iso = _ISO_BY_FLORES.get(source_flores)
            for i, text in zip(indices, translated, strict=True):
                results[i] = TranslationResult(text=text, source_lang=source_iso, translated=True)

        return results  # type: ignore[return-value]  # every slot filled above

    def _translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        self._tokenizer.src_lang = source_lang
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=_MAX_INPUT_TOKENS,
        )
        target_id = self._tokenizer.convert_tokens_to_ids(target_lang)
        generated = self._model.generate(
            **inputs, forced_bos_token_id=target_id, max_length=_MAX_INPUT_TOKENS
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)
