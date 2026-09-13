from typing import Dict, List, Optional, Sequence, Tuple

ISO1_TO_ISO3: Dict[str, str] = {
    "af": "afr", "am": "amh", "ar": "ara", "az": "aze", "be": "bel",
    "bg": "bul", "bn": "ben", "bo": "bod", "bs": "bos", "ca": "cat",
    "cs": "ces", "cy": "cym", "da": "dan", "de": "deu", "el": "ell",
    "en": "eng", "eo": "epo", "es": "spa", "et": "est", "eu": "eus",
    "fa": "fas", "fi": "fin", "fr": "fra", "ga": "gle", "gl": "glg",
    "gu": "guj", "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun",
    "hy": "hye", "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "ka": "kat", "kk": "kaz", "km": "khm", "kn": "kan", "ko": "kor",
    "lo": "lao", "lt": "lit", "lv": "lav", "mk": "mkd", "ml": "mal",
    "mn": "mon", "mr": "mar", "ms": "msa", "my": "mya", "ne": "nep",
    "nl": "nld", "no": "nor", "pa": "pan", "pl": "pol", "pt": "por",
    "ro": "ron", "ru": "rus", "si": "sin", "sk": "slk", "sl": "slv",
    "sq": "sqi", "sr": "srp", "sv": "swe", "sw": "swa", "ta": "tam",
    "te": "tel", "th": "tha", "tl": "tgl", "tr": "tur", "uk": "ukr",
    "ur": "urd", "uz": "uzb", "vi": "vie", "zh": "zho", "zu": "zul",
}

ISO3_TO_ISO1: Dict[str, str] = {v: k for k, v in ISO1_TO_ISO3.items()}

ISO3_ALIASES: Dict[str, str] = {
    "cmn": "zho",
    "yue": "zho",
    "chi": "zho",
    "zh-cn": "zho",
    "zh-tw": "zho",
    "zh_hans": "zho",
    "zh_hant": "zho",
    "nob": "nor",
    "nno": "nor",
    "ger": "deu",
    "fre": "fra",
    "dut": "nld",
    "gre": "ell",
    "per": "fas",
    "cze": "ces",
    "rum": "ron",
    "slo": "slk",
    "arm": "hye",
    "geo": "kat",
    "ice": "isl",
    "may": "msa",
    "alb": "sqi",
    "baq": "eus",
    "wel": "cym",
    "bur": "mya",
    "tib": "bod",
    "mac": "mkd",
    "ind": "ind",
}

LANGUAGE_NAMES: Dict[str, str] = {
    "afr": "Afrikaans", "amh": "Amharic", "ara": "Arabic", "aze": "Azerbaijani",
    "bel": "Belarusian", "ben": "Bengali", "bod": "Tibetan", "bos": "Bosnian",
    "bul": "Bulgarian", "cat": "Catalan", "ces": "Czech", "cym": "Welsh",
    "dan": "Danish", "deu": "German", "ell": "Greek", "eng": "English",
    "epo": "Esperanto", "est": "Estonian", "eus": "Basque", "fas": "Persian",
    "fin": "Finnish", "fra": "French", "gle": "Irish", "glg": "Galician",
    "guj": "Gujarati", "heb": "Hebrew", "hin": "Hindi", "hrv": "Croatian",
    "hun": "Hungarian", "hye": "Armenian", "ind": "Indonesian", "isl": "Icelandic",
    "ita": "Italian", "jpn": "Japanese", "kan": "Kannada", "kat": "Georgian",
    "kaz": "Kazakh", "khm": "Khmer", "kor": "Korean", "lao": "Lao",
    "lav": "Latvian", "lit": "Lithuanian", "mal": "Malayalam", "mar": "Marathi",
    "mkd": "Macedonian", "mon": "Mongolian", "msa": "Malay", "mya": "Burmese",
    "nep": "Nepali", "nld": "Dutch", "nor": "Norwegian", "pan": "Punjabi",
    "pol": "Polish", "por": "Portuguese", "ron": "Romanian", "rus": "Russian",
    "sin": "Sinhala", "slk": "Slovak", "slv": "Slovenian", "spa": "Spanish",
    "sqi": "Albanian", "srp": "Serbian", "swa": "Swahili", "swe": "Swedish",
    "tam": "Tamil", "tel": "Telugu", "tgl": "Tagalog", "tha": "Thai",
    "tur": "Turkish", "ukr": "Ukrainian", "urd": "Urdu", "uzb": "Uzbek",
    "vie": "Vietnamese", "zho": "Chinese", "zul": "Zulu",
}

SCRIPT_RANGES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "Hangul": ((0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F)),
    "Kana": ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)),
    "Han": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)),
    "Latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
    "Cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF)),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Hebrew": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    "Greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "Bengali": ((0x0980, 0x09FF),),
    "Gurmukhi": ((0x0A00, 0x0A7F),),
    "Gujarati": ((0x0A80, 0x0AFF),),
    "Tamil": ((0x0B80, 0x0BFF),),
    "Telugu": ((0x0C00, 0x0C7F),),
    "Kannada": ((0x0C80, 0x0CFF),),
    "Malayalam": ((0x0D00, 0x0D7F),),
    "Sinhala": ((0x0D80, 0x0DFF),),
    "Thai": ((0x0E00, 0x0E7F),),
    "Lao": ((0x0E80, 0x0EFF),),
    "Tibetan": ((0x0F00, 0x0FFF),),
    "Myanmar": ((0x1000, 0x109F),),
    "Georgian": ((0x10A0, 0x10FF), (0x2D00, 0x2D2F)),
    "Armenian": ((0x0530, 0x058F),),
    "Ethiopic": ((0x1200, 0x137F),),
    "Khmer": ((0x1780, 0x17FF),),
}

SCRIPT_ANCHORS: Dict[str, str] = {
    "Hangul": "Korean Hangul characters printed on the document",
    "Kana": "Japanese hiragana and katakana characters printed on the document",
    "Han": "Chinese Han ideographic characters printed on the document",
    "Latin": "Latin alphabet letters printed on the document",
    "Cyrillic": "Cyrillic alphabet letters printed on the document",
    "Arabic": "Arabic script right to left writing printed on the document",
    "Hebrew": "Hebrew script letters printed on the document",
    "Greek": "Greek alphabet letters printed on the document",
    "Devanagari": "Devanagari script with headline bar printed on the document",
    "Bengali": "Bengali script letters printed on the document",
    "Tamil": "Tamil script letters printed on the document",
    "Telugu": "Telugu script letters printed on the document",
    "Thai": "Thai script letters printed on the document",
    "Georgian": "Georgian script letters printed on the document",
    "Armenian": "Armenian script letters printed on the document",
    "Khmer": "Khmer script letters printed on the document",
    "Myanmar": "Burmese Myanmar script letters printed on the document",
}

SCRIPT_CHROME_ANCHORS: Tuple[str, ...] = (
    "company logo emblem graphic",
    "official stamp seal signature mark",
    "blank white margin empty paper area",
    "table border ruling line grid",
    "barcode qr code block",
    "photograph picture illustration",
)

SCRIPT_EXCLUSIVE_LANG: Dict[str, str] = {
    "Hangul": "kor",
    "Kana": "jpn",
    "Thai": "tha",
    "Lao": "lao",
    "Hebrew": "heb",
    "Greek": "ell",
    "Georgian": "kat",
    "Armenian": "hye",
    "Khmer": "khm",
    "Myanmar": "mya",
    "Tamil": "tam",
    "Telugu": "tel",
    "Kannada": "kan",
    "Malayalam": "mal",
    "Sinhala": "sin",
    "Gurmukhi": "pan",
    "Gujarati": "guj",
    "Ethiopic": "amh",
    "Tibetan": "bod",
    "Han": "zho",
    "Bengali": "ben",
}

SCRIPT_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "Hangul": ("kor",),
    "Kana": ("jpn",),
    "Han": ("zho", "jpn"),
    "Latin": (
        "eng", "deu", "fra", "spa", "por", "ita", "nld", "pol", "ces", "slk",
        "hrv", "slv", "ron", "hun", "tur", "vie", "ind", "msa", "tgl", "swe",
        "dan", "nor", "fin", "est", "lav", "lit", "cat", "glg", "eus", "isl",
        "sqi", "afr", "swa", "zul", "cym", "gle", "epo", "uzb", "aze",
    ),
    "Cyrillic": ("rus", "ukr", "bul", "srp", "mkd", "bel", "kaz", "mon"),
    "Arabic": ("ara", "fas", "urd"),
    "Devanagari": ("hin", "mar", "nep"),
    "Bengali": ("ben",),
    "Hebrew": ("heb",),
    "Greek": ("ell",),
    "Thai": ("tha",),
    "Lao": ("lao",),
    "Georgian": ("kat",),
    "Armenian": ("hye",),
    "Khmer": ("khm",),
    "Myanmar": ("mya",),
    "Tamil": ("tam",),
    "Telugu": ("tel",),
    "Kannada": ("kan",),
    "Malayalam": ("mal",),
    "Sinhala": ("sin",),
    "Gurmukhi": ("pan",),
    "Gujarati": ("guj",),
    "Ethiopic": ("amh",),
    "Tibetan": ("bod",),
}

LANG_ANCHOR_TEMPLATE = "a document written in the {name} language"

REFERENCE_LANGUAGE = "eng"

LANG_ANCHOR_TEMPLATES: Tuple[str, ...] = (
    "a document written in the {name} language",
    "this printed text is written in {name}",
    "{name} words sentences and grammar",
    "an official form filled out in {name}",
)

NOISE_ANCHORS: Tuple[str, ...] = (
    "random meaningless characters that belong to no language",
    "machine generated placeholder tokens and markup fragments",
    "html xml tags attributes and special control tokens",
    "a shuffled list of unrelated words without grammar",
    "garbled unreadable optical character recognition output",
)

NOISE_QUALITY_FLOOR = 0.0
REFERENCE_DECISION_MARGIN = 0.020
PEER_DECISION_MARGIN = 0.010
STRONG_REFERENCE_MARGIN = 0.060

CODEPOINT_SIGNATURES: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "deu": ((0x00DF, 0x00DF), (0x00C4, 0x00C4), (0x00D6, 0x00D6), (0x00DC, 0x00DC),
            (0x00E4, 0x00E4), (0x00F6, 0x00F6), (0x00FC, 0x00FC)),
    "fra": ((0x00E7, 0x00E7), (0x00E8, 0x00EA), (0x0153, 0x0153), (0x00F9, 0x00FB)),
    "spa": ((0x00F1, 0x00F1), (0x00BF, 0x00BF), (0x00A1, 0x00A1)),
    "por": ((0x00E3, 0x00E3), (0x00F5, 0x00F5), (0x00E7, 0x00E7)),
    "ita": ((0x00E0, 0x00E0), (0x00EC, 0x00EC), (0x00F2, 0x00F2)),
    "pol": ((0x0141, 0x0144), (0x0179, 0x017C), (0x0104, 0x0107), (0x0118, 0x011B)),
    "ces": ((0x0158, 0x0159), (0x011A, 0x011B), (0x016E, 0x016F), (0x010C, 0x010D)),
    "slk": ((0x013D, 0x013E), (0x0139, 0x013A), (0x0154, 0x0155), (0x00F4, 0x00F4)),
    "hrv": ((0x0106, 0x0107), (0x0110, 0x0111), (0x010C, 0x010D)),
    "slv": ((0x010C, 0x010D), (0x0160, 0x0161), (0x017D, 0x017E)),
    "ron": ((0x0102, 0x0103), (0x0218, 0x021B), (0x00EE, 0x00EE)),
    "hun": ((0x0150, 0x0151), (0x0170, 0x0171)),
    "tur": ((0x011E, 0x011F), (0x0130, 0x0131), (0x015E, 0x015F)),
    "vie": ((0x01A0, 0x01B0), (0x1EA0, 0x1EF9), (0x0110, 0x0111)),
    "swe": ((0x00C5, 0x00C5), (0x00E5, 0x00E5)),
    "dan": ((0x00C6, 0x00C6), (0x00D8, 0x00D8), (0x00E6, 0x00E6), (0x00F8, 0x00F8)),
    "nor": ((0x00C6, 0x00C6), (0x00D8, 0x00D8), (0x00E6, 0x00E6), (0x00F8, 0x00F8)),
    "est": ((0x00F5, 0x00F5), (0x00E4, 0x00E4), (0x00FC, 0x00FC)),
    "lav": ((0x0100, 0x0101), (0x0136, 0x013C), (0x0145, 0x0146)),
    "lit": ((0x0104, 0x0105), (0x0116, 0x0119), (0x0172, 0x0173)),
    "isl": ((0x00DE, 0x00FE), (0x00D0, 0x00F0)),
    "sqi": ((0x00EB, 0x00EB),),
    "ukr": ((0x0404, 0x0404), (0x0406, 0x0407), (0x0454, 0x0457), (0x0490, 0x0491)),
    "bel": ((0x040E, 0x040E), (0x045E, 0x045E), (0x0406, 0x0406)),
    "bul": ((0x044A, 0x044A), (0x0429, 0x0429)),
    "srp": ((0x0402, 0x0402), (0x040B, 0x040B), (0x0409, 0x040A), (0x045F, 0x045F)),
    "mkd": ((0x0403, 0x0403), (0x040C, 0x040C), (0x0405, 0x0405)),
    "kaz": ((0x04D8, 0x04D9), (0x0492, 0x0493), (0x049A, 0x049B), (0x04B0, 0x04B3)),
    "mon": ((0x04E8, 0x04E9), (0x04AE, 0x04AF)),
    "fas": ((0x067E, 0x067E), (0x0686, 0x0686), (0x0698, 0x0698), (0x06AF, 0x06AF)),
    "urd": ((0x0679, 0x0679), (0x0688, 0x0688), (0x0691, 0x0691), (0x06BE, 0x06BE)),
    "jpn": ((0x3040, 0x30FF),),
    "kor": ((0xAC00, 0xD7A3), (0x1100, 0x11FF)),
}

LATIN_PRIORS: Dict[str, float] = {
    "eng": 1.00, "deu": 0.62, "fra": 0.62, "spa": 0.62, "por": 0.58,
    "ita": 0.56, "nld": 0.52, "pol": 0.48, "tur": 0.48, "vie": 0.48,
    "ind": 0.46, "msa": 0.42, "ces": 0.42, "swe": 0.40, "dan": 0.38,
    "nor": 0.38, "fin": 0.38, "hun": 0.38, "ron": 0.38, "hrv": 0.34,
    "slk": 0.34, "slv": 0.32, "cat": 0.32, "tgl": 0.32, "est": 0.28,
    "lav": 0.28, "lit": 0.28, "glg": 0.26, "eus": 0.24, "isl": 0.24,
    "sqi": 0.24, "afr": 0.24, "swa": 0.24, "zul": 0.20, "cym": 0.20,
    "gle": 0.20, "epo": 0.16, "uzb": 0.24, "aze": 0.24,
}

CYRILLIC_PRIORS: Dict[str, float] = {
    "rus": 1.00, "ukr": 0.66, "bul": 0.50, "srp": 0.48,
    "mkd": 0.38, "bel": 0.36, "kaz": 0.36, "mon": 0.30,
}

ARABIC_PRIORS: Dict[str, float] = {
    "ara": 1.00, "fas": 0.62, "urd": 0.52,
}

DEVANAGARI_PRIORS: Dict[str, float] = {
    "hin": 1.00, "mar": 0.52, "nep": 0.40,
}

HAN_PRIORS: Dict[str, float] = {
    "zho": 1.00, "jpn": 0.40,
}

HANGUL_PRIORS: Dict[str, float] = {
    "kor": 1.00, "jpn": 0.10, "zho": 0.05,
}

KANA_PRIORS: Dict[str, float] = {
    "jpn": 1.00, "kor": 0.15, "zho": 0.05,
}

SCRIPT_PRIORS: Dict[str, Dict[str, float]] = {
    "Latin": LATIN_PRIORS,
    "Cyrillic": CYRILLIC_PRIORS,
    "Arabic": ARABIC_PRIORS,
    "Devanagari": DEVANAGARI_PRIORS,
    "Han": HAN_PRIORS,
    "Hangul": HANGUL_PRIORS,
    "Kana": KANA_PRIORS,
}

DECISIVE_BLOCKS: Tuple[Tuple[str, str, Tuple[Tuple[int, int], ...]], ...] = (
    ("Hangul", "kor", (
        (0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F),
    )),
    ("Kana", "jpn", (
        (0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF),
    )),
    ("Thai", "tha", ((0x0E00, 0x0E7F),)),
    ("Cyrillic", "rus", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("Arabic", "ara", (
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
    )),
    ("Devanagari", "hin", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("Bengali", "ben", ((0x0980, 0x09FF),)),
    ("Greek", "ell", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("Hebrew", "heb", ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))),
    ("Armenian", "hye", ((0x0530, 0x058F),)),
    ("Georgian", "kat", ((0x10A0, 0x10FF), (0x2D00, 0x2D2F))),
    ("Tamil", "tam", ((0x0B80, 0x0BFF),)),
    ("Telugu", "tel", ((0x0C00, 0x0C7F),)),
    ("Kannada", "kan", ((0x0C80, 0x0CFF),)),
    ("Malayalam", "mal", ((0x0D00, 0x0D7F),)),
    ("Sinhala", "sin", ((0x0D80, 0x0DFF),)),
    ("Gurmukhi", "pan", ((0x0A00, 0x0A7F),)),
    ("Gujarati", "guj", ((0x0A80, 0x0AFF),)),
    ("Lao", "lao", ((0x0E80, 0x0EFF),)),
    ("Khmer", "khm", ((0x1780, 0x17FF),)),
    ("Myanmar", "mya", ((0x1000, 0x109F),)),
    ("Ethiopic", "amh", ((0x1200, 0x137F),)),
    ("Tibetan", "bod", ((0x0F00, 0x0FFF),)),
    ("Vietnamese", "vie", ((0x1EA0, 0x1EF9), (0x01A0, 0x01B0))),
)

LATE_BLOCKS: Tuple[Tuple[str, str, Tuple[Tuple[int, int], ...]], ...] = (
    ("Han", "zho", (
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
    )),
)

LATIN_BLOCKS: Tuple[Tuple[int, int], ...] = (
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF),
)

DECISIVE_MIN_CHARS = 1


def _in_blocks(cp: int, blocks: Tuple[Tuple[int, int], ...]) -> bool:
    for lo, hi in blocks:
        if lo <= cp <= hi:
            return True
    return False


def block_census(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ch in str(text or ""):
        if ch.isspace() or ch.isdigit():
            continue
        cp = ord(ch)
        hit = ""
        for name, _code, blocks in DECISIVE_BLOCKS:
            if _in_blocks(cp, blocks):
                hit = name
                break
        if not hit:
            for name, _code, blocks in LATE_BLOCKS:
                if _in_blocks(cp, blocks):
                    hit = name
                    break
        if not hit and _in_blocks(cp, LATIN_BLOCKS):
            hit = "Latin"
        if hit:
            out[hit] = out.get(hit, 0) + 1
    return out


def decide_script_language(
    text: str,
    min_chars: int = DECISIVE_MIN_CHARS,
) -> Tuple[str, str, str]:
    census = block_census(text)
    if not census:
        return "", "", "식별 가능한 문자가 없습니다."

    for name, code, _blocks in DECISIVE_BLOCKS:
        cnt = census.get(name, 0)
        if cnt >= int(min_chars):
            return code, name, (
                f"'{name}' 블록 {cnt}자 검출 — 해당 문자를 쓰는 언어는 "
                f"'{code}' 뿐이므로 즉시 확정"
            )

    for name, code, _blocks in LATE_BLOCKS:
        cnt = census.get(name, 0)
        if cnt >= int(min_chars):
            return code, name, (
                f"'{name}' 블록 {cnt}자 검출 — 한중일 공용 문자이나 "
                f"다른 결정적 블록이 없어 '{code}' 로 확정"
            )

    if census.get("Latin", 0) >= int(min_chars):
        return "", "Latin", (
            f"라틴 알파벳 {census['Latin']}자 — 기능어 프로파일로 "
            f"세부 언어를 가립니다."
        )

    return "", "", f"결정적 블록이 없습니다 ({census})."


LATIN_DEFAULT = "eng"

LATIN_ALLOWED: Tuple[str, ...] = (
    "eng", "fra", "deu", "spa", "ita", "por", "nld",
)

LATIN_PROFILES: Dict[str, Tuple[str, ...]] = {
    "eng": (
        "the", "of", "and", "to", "in", "is", "for", "on", "with", "as",
        "by", "at", "from", "this", "that", "are", "be", "or", "not",
        "was", "have", "has", "it", "an", "all", "which", "shall",
        "invoice", "number", "date", "total", "amount", "country",
        "shipment", "goods", "value", "quantity", "price", "weight",
        "description", "address", "name", "port", "terms", "payment",
    ),
    "fra": (
        "le", "la", "les", "de", "des", "du", "et", "un", "une", "est",
        "que", "qui", "dans", "pour", "sur", "par", "avec", "au", "aux",
        "ne", "pas", "ce", "cette", "sont", "plus", "facture", "numero",
        "date", "montant", "pays", "marchandises", "poids", "adresse",
        "quantite", "prix", "total", "expedition", "paiement",
    ),
    "deu": (
        "der", "die", "das", "und", "ist", "von", "zu", "den", "dem",
        "mit", "auf", "fur", "als", "nicht", "ein", "eine", "einer",
        "sich", "auch", "sind", "wird", "werden", "rechnung", "nummer",
        "datum", "betrag", "land", "waren", "gewicht", "anschrift",
        "menge", "preis", "gesamt", "versand", "zahlung",
    ),
    "spa": (
        "el", "la", "los", "las", "de", "del", "y", "en", "que", "un",
        "una", "por", "con", "para", "es", "son", "no", "se", "su",
        "como", "mas", "factura", "numero", "fecha", "importe", "pais",
        "mercancias", "peso", "direccion", "cantidad", "precio",
        "total", "envio", "pago",
    ),
    "ita": (
        "il", "lo", "la", "i", "gli", "le", "di", "del", "della", "e",
        "in", "che", "un", "una", "per", "con", "non", "si", "sono",
        "come", "piu", "da", "fattura", "numero", "data", "importo",
        "paese", "merci", "peso", "indirizzo", "quantita", "prezzo",
        "totale", "spedizione", "pagamento",
    ),
    "por": (
        "o", "a", "os", "as", "de", "do", "da", "dos", "das", "e",
        "em", "que", "um", "uma", "por", "com", "para", "nao", "se",
        "sao", "mais", "fatura", "numero", "data", "valor", "pais",
        "mercadorias", "peso", "endereco", "quantidade", "preco",
        "total", "remessa", "pagamento",
    ),
    "nld": (
        "de", "het", "een", "van", "en", "is", "in", "op", "te", "dat",
        "die", "voor", "met", "aan", "niet", "zijn", "worden", "als",
        "door", "factuur", "nummer", "datum", "bedrag", "land",
        "goederen", "gewicht", "adres", "aantal", "prijs", "totaal",
        "zending", "betaling",
    ),
}

LATIN_MIN_TOKENS = 6
LATIN_MIN_HITS = 2
LATIN_MIN_MARGIN = 2


def _latin_tokens(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in str(text or ""):
        cp = ord(ch)
        if _in_blocks(cp, LATIN_BLOCKS):
            buf.append(ch.lower())
            continue
        if buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [t for t in out if len(t) >= 1]


def _fold(token: str) -> str:
    table = {
        "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "í": "i", "ì": "i", "î": "i", "ï": "i",
        "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o",
        "ú": "u", "ù": "u", "û": "u", "ü": "u",
        "ç": "c", "ñ": "n", "ß": "s",
    }
    return "".join(table.get(c, c) for c in token)


def decide_latin_language(
    text: str,
    served: Optional[Sequence[str]] = None,
) -> Tuple[str, str, Dict[str, int]]:
    tokens = [_fold(t) for t in _latin_tokens(text)]
    scores: Dict[str, int] = {}

    allowed = list(LATIN_ALLOWED)
    if served:
        extra = [c for c in served if c in LATIN_PROFILES and c not in allowed]
        allowed.extend(extra)

    for code in allowed:
        bank = set(LATIN_PROFILES.get(code, ()))
        if not bank:
            continue
        scores[code] = sum(1 for t in tokens if t in bank)

    if len(tokens) < LATIN_MIN_TOKENS:
        return LATIN_DEFAULT, (
            f"라틴 토큰 {len(tokens)}개는 프로파일 판별에 부족합니다 "
            f"→ 기본 '{LATIN_DEFAULT}'"
        ), scores

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] < LATIN_MIN_HITS:
        return LATIN_DEFAULT, (
            f"어떤 언어 기능어도 {LATIN_MIN_HITS}회 이상 맞지 않습니다 "
            f"→ 기본 '{LATIN_DEFAULT}'"
        ), scores

    top_code, top_hits = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0

    if top_hits - second < LATIN_MIN_MARGIN:
        return LATIN_DEFAULT, (
            f"1위 '{top_code}'({top_hits}) 와 2위({second}) 격차가 "
            f"{LATIN_MIN_MARGIN} 미만 → 기본 '{LATIN_DEFAULT}'"
        ), scores

    return top_code, (
        f"기능어 적중 '{top_code}' {top_hits}회 vs 차점 {second}회 "
        f"(토큰 {len(tokens)}개) → '{top_code}' 확정"
    ), scores


def normalize_lang_code(code: Optional[str]) -> str:
    if not code:
        return "eng"
    c = str(code).strip().lower().replace("-", "_")
    if c in ISO3_ALIASES:
        return ISO3_ALIASES[c]
    if "_" in c:
        head = c.split("_", 1)[0]
        if head in ISO1_TO_ISO3:
            return ISO1_TO_ISO3[head]
        if head in LANGUAGE_NAMES:
            return head
        c = head
    if c in ISO1_TO_ISO3:
        return ISO1_TO_ISO3[c]
    if c in LANGUAGE_NAMES:
        return c
    return c


def language_name(code: str) -> str:
    c = normalize_lang_code(code)
    return LANGUAGE_NAMES.get(c, c)


def iso1_of(code: str) -> str:
    c = normalize_lang_code(code)
    return ISO3_TO_ISO1.get(c, c)


def char_script(ch: str) -> Optional[str]:
    cp = ord(ch)
    for script, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return None


def script_histogram(text: str) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        s = char_script(ch)
        if s is None:
            continue
        hist[s] = hist.get(s, 0) + 1
    return hist


CJK_SCRIPTS = ("Hangul", "Kana", "Han")

SCRIPT_TIE_RATIO = 0.25


def dominant_script(text: str) -> Tuple[Optional[str], float, Dict[str, int]]:
    hist = script_histogram(text)
    if not hist:
        return None, 0.0, hist
    total = sum(hist.values())
    if total <= 0:
        return None, 0.0, hist

    _code, script, _why = decide_script_language(text)
    if script and script in hist:
        return script, hist[script] / total, hist

    ranked = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_cnt = ranked[0]
    return best_name, best_cnt / total, hist


def script_is_decisive(hist: Dict[str, int]) -> Tuple[bool, str]:
    if not hist:
        return False, "문자 표본이 없습니다."
    total = sum(hist.values())
    if total < 8:
        return False, f"문자 표본이 {total}자뿐입니다."

    ranked = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_cnt = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0

    if top_cnt < total * 0.60:
        return False, (
            f"최다 스크립트 '{top_name}' 비중 {top_cnt / total:.0%} < 60%"
        )
    if second and second >= top_cnt * (1.0 - SCRIPT_TIE_RATIO):
        return False, (
            f"'{top_name}'({top_cnt}) vs '{ranked[1][0]}'({second}) 실질 동률"
        )
    return True, ""


CJK_CROSS_CANDIDATES: Tuple[str, ...] = ("kor", "jpn", "zho")


def candidates_for_script(script: Optional[str]) -> List[str]:
    if not script:
        return ["eng"]
    base = list(SCRIPT_CANDIDATES.get(script, ("eng",)))
    if script in CJK_SCRIPTS:
        for c in CJK_CROSS_CANDIDATES:
            if c not in base:
                base.append(c)
    return base


def priors_for_script(script: Optional[str]) -> Dict[str, float]:
    if not script:
        return {"eng": 1.0}
    if script in SCRIPT_PRIORS:
        return dict(SCRIPT_PRIORS[script])
    excl = SCRIPT_EXCLUSIVE_LANG.get(script)
    if excl:
        return {excl: 1.0}
    return {"eng": 1.0}


NON_EXCLUSIVE_SCRIPTS = (
    "Han", "Latin", "Cyrillic", "Arabic", "Devanagari", "Hangul", "Kana",
)


def exclusive_language_of(script: Optional[str]) -> Optional[str]:
    if not script:
        return None
    if script in NON_EXCLUSIVE_SCRIPTS:
        return None
    return SCRIPT_EXCLUSIVE_LANG.get(script)


def lang_anchor_phrases(code: str) -> List[str]:
    name = language_name(code)
    return [tpl.format(name=name) for tpl in LANG_ANCHOR_TEMPLATES]


def reference_language_of(script: Optional[str]) -> str:
    excl = SCRIPT_EXCLUSIVE_LANG.get(script or "")
    if excl:
        return excl
    priors = priors_for_script(script)
    if not priors:
        return REFERENCE_LANGUAGE
    return max(priors.items(), key=lambda kv: kv[1])[0]