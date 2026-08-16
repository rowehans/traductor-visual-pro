"""Evaluación multilingüe y explicable de traducciones OCR.

El evaluador no intenta sustituir una revisión humana. Su objetivo es separar
fallos reales de OCR/traducción de casos válidos que no deben traducirse,
como SFX o nombres preservados, y reportar cada combinación de idiomas por
separado.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Any, Iterable


LANG_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "spa": "es",
    "eng": "en",
    "jpn": "ja",
    "kor": "ko",
}

SFX_TYPES = {
    "sfx", "sound", "sound_effect", "onomatopoeia", "onomatopeya",
}
NAME_TYPES = {
    "name", "proper_name", "character", "character_name", "person",
    "person_name",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
_SCRIPT_RE = {
    "ru": re.compile(r"[А-Яа-яЁё]"),
    "uk": re.compile(r"[А-Яа-яІіЇїЄєҐґ]"),
    "bg": re.compile(r"[А-Яа-я]"),
    "ar": re.compile(r"[\u0600-\u06ff]"),
    "fa": re.compile(r"[\u0600-\u06ff]"),
    "he": re.compile(r"[\u0590-\u05ff]"),
    "el": re.compile(r"[\u0370-\u03ff]"),
    "hi": re.compile(r"[\u0900-\u097f]"),
    "bn": re.compile(r"[\u0980-\u09ff]"),
    "th": re.compile(r"[\u0e00-\u0e7f]"),
    "ka": re.compile(r"[\u10a0-\u10ff]"),
    "hy": re.compile(r"[\u0530-\u058f]"),
}

_COMMON_WORDS = {
    "es": {"el", "la", "los", "las", "un", "una", "de", "en", "que", "y", "no", "sí", "si", "para", "con", "hola", "mundo"},
    "en": {"the", "a", "an", "of", "in", "that", "and", "is", "are", "not", "for", "with", "hello", "world"},
    "fr": {"le", "la", "les", "un", "une", "de", "des", "et", "est", "pas", "bonjour"},
    "de": {"der", "die", "das", "ein", "eine", "und", "ist", "nicht", "hallo"},
    "it": {"il", "lo", "la", "gli", "un", "una", "di", "e", "è", "non", "ciao"},
    "pt": {"o", "a", "os", "as", "um", "uma", "de", "e", "é", "não", "olá"},
}

_KNOWN_SFX = {
    "bang", "boom", "crash", "click", "clack", "slam", "smash", "splash",
    "tap", "knock", "thud", "thump", "sigh", "gasp", "hiss", "buzz",
    "ring", "ding", "dong", "pop", "zap", "pow", "kaboom", "pum", "zas",
    "don", "doki", "zudon", "zuban", "pachin", "grrr", "ah", "oh", "eh",
}


def normalize_lang(value: Any, default: str = "auto") -> str:
    code = str(value or default).strip().lower().replace("_", "-")
    return LANG_ALIASES.get(code, code or default)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _same_text(source: str, target: str) -> bool:
    def normalized(value: str) -> str:
        return re.sub(r"[\s\"'¿¡!?.,;:()\[\]{}_-]+", "", value).casefold()

    return normalized(source) == normalized(target)


def _is_ocr_garbage(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if _CJK_RE.search(text) or _KANA_RE.search(text) or _HANGUL_RE.search(text):
        return False
    if re.fullmatch(r"[\d\s.,;:'\"\-_~]+", text):
        return True
    if re.search(r"\d[A-Z]{5,}|[A-Z]{6,}\d|\d{2,}[A-Z]{2,}\d{2,}", text):
        return True
    digits = sum(char.isdigit() for char in text)
    if digits and digits / max(len(text), 1) > 0.4:
        return True
    if len(re.findall(r"[@#$%^&*+=<>\[\]{}|\\/]", text)) >= 3:
        return True
    letters = "".join(char for char in text if char.isalpha())
    if len(letters) <= 2 and not re.search(r"[aeiouAEIOU]", letters) and letters.isascii():
        return True
    return False


def _is_sfx(value: str, block_type: str = "") -> bool:
    kind = block_type.strip().lower().replace("-", "_")
    if kind in SFX_TYPES:
        return True
    stripped = value.strip(" '!?.,;:~-_()[]{}")
    if not stripped or len(stripped) > 20:
        return False
    lowered = stripped.casefold()
    if lowered in _KNOWN_SFX:
        return True
    if re.search(r"(.)\1{2,}", lowered):
        return True
    if lowered.startswith("*") and lowered.endswith("*"):
        return True
    return False


def _looks_like_language(value: str, language: str) -> bool:
    lang = normalize_lang(language)
    if lang in {"ja"}:
        return bool(_KANA_RE.search(value))
    if lang in {"ko"}:
        return bool(_HANGUL_RE.search(value))
    if lang in {"zh"}:
        return bool(_CJK_RE.search(value)) and not bool(_KANA_RE.search(value))
    script = _SCRIPT_RE.get(lang)
    if script is not None:
        return bool(script.search(value))
    if lang in _COMMON_WORDS:
        words = {
            word.casefold().strip(".,;:!?¿¡()[]{}\"'")
            for word in value.split()
        }
        if words & _COMMON_WORDS[lang]:
            return True
    # Para idiomas con alfabeto latino no basta con encontrar letras: eso
    # convertiría cualquier nombre, ruido OCR o palabra desconocida en
    # "idioma destino". Exigimos una palabra frecuente o un idioma declarado
    # explícitamente (el caso source_lang == target_lang se resuelve antes).
    return False


def _is_name_type(block_type: str) -> bool:
    return block_type.strip().lower().replace("-", "_") in NAME_TYPES


@dataclass(frozen=True)
class PairQuality:
    source: str
    target: str
    category: str
    accepted: bool
    reason: str = ""


def classify_pair(
    source: Any,
    target: Any,
    *,
    source_lang: Any = "auto",
    target_lang: Any = "auto",
    block_type: Any = "",
) -> PairQuality:
    """Clasifica un par sin depender de modelos ni de red."""
    src = _text(source)
    tgt = _text(target)
    src_lang = normalize_lang(source_lang)
    tgt_lang = normalize_lang(target_lang)
    kind = _text(block_type).lower()

    if not src and not tgt:
        return PairQuality(src, tgt, "EMPTY", False, "ambos textos están vacíos")
    if _is_sfx(src, kind):
        if _same_text(src, tgt):
            return PairQuality(src, tgt, "SFX_PRESERVED", True, "efecto de sonido preservado")
        if tgt and not _is_ocr_garbage(tgt):
            return PairQuality(src, tgt, "SFX_TRANSLATED", True, "efecto de sonido traducido")

    source_garbage = _is_ocr_garbage(src)
    target_garbage = _is_ocr_garbage(tgt)
    if source_garbage:
        if tgt and not _same_text(src, tgt) and not target_garbage:
            return PairQuality(src, tgt, "OCR_NOISY_RECOVERED", True, "OCR ruidoso recuperado")
        return PairQuality(src, tgt, "OCR_GARBAGE", False, "fuente con señales fuertes de OCR basura")

    if _is_name_type(kind) and _same_text(src, tgt):
        return PairQuality(src, tgt, "NAME_PRESERVED", True, "nombre propio preservado")

    if _same_text(src, tgt):
        if src_lang == tgt_lang and tgt_lang != "auto":
            return PairQuality(src, tgt, "ALREADY_TARGET", True, "origen y destino coinciden")
        if tgt_lang != "auto" and _looks_like_language(src, tgt_lang):
            return PairQuality(src, tgt, "ALREADY_TARGET", True, "texto ya está en el idioma destino")
        return PairQuality(src, tgt, "UNTRANSLATED", False, "texto idéntico sin contexto suficiente")

    if not tgt:
        return PairQuality(src, tgt, "UNTRANSLATED", False, "destino vacío")
    if target_garbage:
        return PairQuality(src, tgt, "BAD_TRANSLATION", False, "destino con señales de OCR basura")

    words = [word for word in re.findall(r"\w+", tgt, flags=re.UNICODE) if word]
    source_words = [word for word in re.findall(r"\w+", src, flags=re.UNICODE) if word]
    ratio = len(words) / max(len(source_words), 1)
    if len(source_words) >= 3 and len(words) >= 3 and 0.5 < ratio < 1.5:
        return PairQuality(src, tgt, "LITERAL_TRANSLATION", True, "longitud comparable")
    if tgt_lang == "auto" or _looks_like_language(tgt, tgt_lang):
        return PairQuality(src, tgt, "GOOD_TRANSLATION", True, "destino compatible con el idioma declarado")
    return PairQuality(src, tgt, "REVIEW_LANGUAGE", False, "destino no coincide claramente con el idioma declarado")


@dataclass
class QualityBucket:
    total: int = 0
    accepted: int = 0
    categories: Counter[str] = field(default_factory=Counter)

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.total * 100.0) if self.total else 0.0


@dataclass
class QualityReport:
    total: int = 0
    accepted: int = 0
    metadata_items: int = 0
    by_pair: dict[str, QualityBucket] = field(default_factory=dict)
    categories: Counter[str] = field(default_factory=Counter)
    consistency_conflicts: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def metadata_coverage(self) -> float:
        return self.metadata_items / self.total if self.total else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total * 100.0 if self.total else 0.0

    @property
    def consistency_conflict_count(self) -> int:
        return len(self.consistency_conflicts)


def _pair_key(source_lang: Any, target_lang: Any) -> str:
    return f"{normalize_lang(source_lang)}→{normalize_lang(target_lang)}"


def analyze_checkpoint(
    checkpoint: dict[str, Any],
    *,
    source_lang: Any | None = None,
    target_lang: Any | None = None,
) -> QualityReport:
    """Analiza un checkpoint antiguo o uno con metadatos completos."""
    default_source = source_lang if source_lang is not None else checkpoint.get("source_lang", "auto")
    default_target = target_lang if target_lang is not None else checkpoint.get("target_lang", "auto")
    report = QualityReport()
    variants: dict[tuple[str, str, str], tuple[str, set[str]]] = {}

    for page in checkpoint.get("results", []):
        if not isinstance(page, dict):
            continue
        for item in page.get("texts", []):
            if not isinstance(item, dict):
                continue
            item_source = item.get("src", item.get("source", ""))
            item_target = item.get("tgt", item.get("translated", ""))
            item_source_lang = item.get("source_lang", page.get("source_lang", default_source))
            item_target_lang = item.get("target_lang", page.get("target_lang", default_target))
            block_type = item.get("type", item.get("block_type", ""))
            has_metadata = bool(item.get("type") or item.get("block_type") or item.get("source_lang") or item.get("target_lang"))
            quality = classify_pair(
                item_source,
                item_target,
                source_lang=item_source_lang,
                target_lang=item_target_lang,
                block_type=block_type,
            )
            key = _pair_key(item_source_lang, item_target_lang)
            bucket = report.by_pair.setdefault(key, QualityBucket())
            bucket.total += 1
            bucket.accepted += int(quality.accepted)
            bucket.categories[quality.category] += 1
            report.total += 1
            report.accepted += int(quality.accepted)
            report.categories[quality.category] += 1
            report.metadata_items += int(has_metadata)

            # La consistencia se mide solo sobre texto real y no sobre SFX o
            # OCR basura. El par de idiomas forma parte de la clave para no
            # mezclar la misma cadena entre ejecuciones distintas.
            if (
                item_source and item_target and not _is_sfx(item_source, _text(block_type))
                and quality.category not in {"OCR_GARBAGE", "EMPTY"}
            ):
                normalized_source = re.sub(r"\s+", " ", _text(item_source)).casefold()
                variant_key = (key, normalized_source, _text(block_type).lower())
                original, targets = variants.setdefault(
                    variant_key, (_text(item_source), set())
                )
                targets.add(_text(item_target))

    for (pair, _source_key, block_type), (original, targets) in variants.items():
        normalized_targets = {target.casefold() for target in targets}
        if len(normalized_targets) > 1:
            report.consistency_conflicts[f"{pair}:{block_type}:{original}"] = tuple(sorted(targets))
    return report


def render_report(report: QualityReport) -> str:
    """Renderiza un informe estable para humanos y para el runner CI."""
    lines = [
        "=" * 90,
        f"  INFORME MULTILINGÜE DE CALIDAD — {report.total} pares analizados",
        "=" * 90,
        f"  Tasa de aceptacion global: {report.acceptance_rate:.1f}%",
        f"  Cobertura de metadatos semánticos: {report.metadata_coverage * 100:.1f}%",
        f"  Conflictos de consistencia: {report.consistency_conflict_count}",
        "",
        "  POR IDIOMA",
    ]
    for pair, bucket in sorted(report.by_pair.items()):
        lines.append(
            f"  {pair}: {bucket.total} pares | {bucket.acceptance_rate:.1f}% aceptables | "
            f"{dict(sorted(bucket.categories.items()))}"
        )
    lines.extend(["", "  CATEGORÍAS"])
    for category, count in sorted(report.categories.items()):
        lines.append(f"  {category}: {count}")
    return "\n".join(lines) + "\n"
