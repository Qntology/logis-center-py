from typing import Dict, List, Optional, Tuple

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

SCRIPT_PRIORS: Dict[str, Dict[str, float]] = {
    "Latin": LATIN_PRIORS,
    "Cyrillic": CYRILLIC_PRIORS,
    "Arabic": ARABIC_PRIORS,
    "Devanagari": DEVANAGARI_PRIORS,
    "Han": HAN_PRIORS,
}


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


def dominant_script(text: str) -> Tuple[Optional[str], float, Dict[str, int]]:
    hist = script_histogram(text)
    if not hist:
        return None, 0.0, hist
    total = sum(hist.values())
    if "Kana" in hist and hist["Kana"] >= max(2, int(total * 0.02)):
        return "Kana", hist["Kana"] / total, hist
    if "Hangul" in hist and hist["Hangul"] >= max(2, int(total * 0.05)):
        return "Hangul", hist["Hangul"] / total, hist
    best = max(hist.items(), key=lambda kv: kv[1])
    return best[0], best[1] / total, hist


def candidates_for_script(script: Optional[str]) -> List[str]:
    if not script:
        return ["eng"]
    return list(SCRIPT_CANDIDATES.get(script, ("eng",)))


def priors_for_script(script: Optional[str]) -> Dict[str, float]:
    if not script:
        return {"eng": 1.0}
    if script in SCRIPT_PRIORS:
        return dict(SCRIPT_PRIORS[script])
    excl = SCRIPT_EXCLUSIVE_LANG.get(script)
    if excl:
        return {excl: 1.0}
    return {"eng": 1.0}


def exclusive_language_of(script: Optional[str]) -> Optional[str]:
    if not script:
        return None
    if script in ("Han", "Latin", "Cyrillic", "Arabic", "Devanagari"):
        return None
    return SCRIPT_EXCLUSIVE_LANG.get(script)