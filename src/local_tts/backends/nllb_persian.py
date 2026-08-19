"""Dedicated local English-to-Persian translation through NLLB-200."""

from __future__ import annotations

import gc
import re
import time

from ..models import ResourceSettings
from ..summarize import SummaryResult


DEFAULT_NLLB_MODEL = "facebook/nllb-200-distilled-600M"


class NLLBPersianTranslator:
    """A literal neural translator, run locally on MPS when it is available.

    Unlike a general chat model, NLLB does not follow an instruction prompt or
    invent a response around the source. This makes it the safer default for
    source-preserving book translation. It is intentionally a separate phase
    from Piper so their unified-memory allocations never overlap.
    """

    name = "nllb-mps"

    def __init__(self, model: str = DEFAULT_NLLB_MODEL) -> None:
        self.model_name = model
        self._model = None
        self._tokenizer = None
        self._device = None

    def configure(self, resources: ResourceSettings) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - installation concern
            raise RuntimeError("NLLB translation support is unavailable; reinstall the project dependencies") from exc
        torch.set_num_threads(resources.cpu_threads)
        self._device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def translate_to_persian(
        self,
        source_text: str,
        *,
        max_tokens: int,
        glossary: tuple[tuple[str, str], ...] = (),
    ) -> SummaryResult:
        if not source_text.strip():
            raise ValueError("cannot translate an empty text segment")
        if self._device is None:
            raise RuntimeError("NLLB translator must be configured before use")
        model, tokenizer = self._load()
        import torch

        protected_source, protected_terms = _protect_glossary(source_text, glossary)
        inputs = tokenizer(
            protected_source,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("pes_Arab"),
                # NLLB ships a legacy generation_config with max_length=200.
                # Passing max_new_tokens alongside it emits one warning per
                # chunk. A decoder-length limit expresses the same cap without
                # conflicting with that inherited setting.
                max_length=_decoder_max_length(model, max_tokens),
                num_beams=4,
            )
        if self._device.type == "mps":
            torch.mps.synchronize()
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        text = _restore_glossary(text, protected_terms)
        if not text:
            raise RuntimeError("the local NLLB model produced no Persian text")
        # SummaryResult is retained as the common generation metric value for
        # the first translation backend. The zero MLX values explicitly mean
        # this record came from NLLB/Torch rather than MLX.
        return SummaryResult(
            text=text,
            model=self.model_name,
            input_tokens=int(inputs["input_ids"].shape[-1]),
            output_tokens=int(generated.shape[-1]),
            wall_seconds=time.perf_counter() - started,
            mlx_active_mb=0.0,
            mlx_peak_mb=0.0,
        )

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - installation concern
            raise RuntimeError("NLLB translation support is unavailable; reinstall the project dependencies") from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="eng_Latn", local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
        except OSError:
            # This is an explicit first-use model download. Later calls resolve
            # from the Hugging Face cache without network access.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="eng_Latn")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._tokenizer = tokenizer
        self._model = model.to(self._device).eval()
        return self._model, self._tokenizer

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:  # pragma: no cover - installation concern
            pass


def _protect_glossary(source_text: str, glossary: tuple[tuple[str, str], ...]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Give NLLB exact Persian terminology without relying on prompt following.

    NLLB consistently copies bracketed tokens. Replacing source terms with a
    unique token before direct translation, then restoring the supplied Persian
    target afterward, preserves the caller's glossary without altering source
    code, URLs, or unrelated text.
    """
    protected = source_text
    restored: list[tuple[str, str]] = []
    for index, (source, target) in enumerate(sorted(glossary, key=lambda item: len(item[0]), reverse=True), start=1):
        if not source.strip() or not target.strip():
            continue
        placeholder = f"[[LOCAL_TTS_TERM_{index:03d}]]"
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
        protected, substitutions = pattern.subn(placeholder, protected)
        if substitutions:
            restored.append((placeholder, target))
    return protected, tuple(restored)


def _restore_glossary(text: str, protected_terms: tuple[tuple[str, str], ...]) -> str:
    """Restore NLLB's copied token despite its harmless bracket normalization."""
    restored = text
    for placeholder, target in protected_terms:
        bare = placeholder[2:-2]
        for variant in (placeholder, f"[{bare}]", bare):
            restored = restored.replace(variant, target)
    return restored


def _decoder_max_length(model, max_new_tokens: int) -> int:
    """Translate a new-token budget into NLLB's decoder sequence limit."""
    requested = max_new_tokens + 1  # one initial decoder token precedes output
    model_limit = getattr(model.config, "max_position_embeddings", None)
    return min(requested, int(model_limit)) if model_limit else requested
