"""
Bilingual NLP Emotion Analyzer (Hindi & English)
Analyzes speech transcripts or text semantics in both Hindi (Devanagari & Romanized/Hinglish)
and English using affective lexicon mapping, valence-arousal scoring, and negation parsing.
"""

import re
import unicodedata
from src.config import EMOTIONS

# Hindi Emotional Lexicon (Devanagari & Hinglish)
HINDI_EMOTION_LEXICON = {
    "Angry": {
        "words": [
            "गुस्सा", "क्रोध", "नफरत", "पागल", "बकवास", "झगड़ा", "मारूंगा", "कमीने", "चिल्लाना",
            "बेवकूफ", "बकवास", "बदतमीज", "चुप", "भड़कना", "धोखेबाज", "गाली", "तंग", "आक्रोश",
            "gussa", "krodh", "pagal", "bakwas", "marunga", "chup", "chilla", "bewakoof", "bevkoof", "tang"
        ],
        "weight": 1.2
    },
    "Happy": {
        "words": [
            "खुश", "खुशी", "बधाई", "शानदार", "मजा", "प्यार", "खूबसूरत", "जीत", "उत्सव", "धन्यवाद",
            "आनंद", "मुस्कान", "वाह", "कमाल", "बढ़िया", "शुक्रिया", "रोमांचक", "बेहतरीन", "प्यारा",
            "khush", "khushi", "badhai", "shandar", "maza", "pyar", "khoobsurat", "jeet", "wah", "kamaal", "badhiya"
        ],
        "weight": 1.1
    },
    "Sad": {
        "words": [
            "उदास", "उदासी", "रोना", "दुख", "गम", "तन्हा", "दर्द", "मायूस", "अकेला", "अफ़सोस",
            "टूटना", "निराशा", "बेचैन", "मातम", "तकलीफ", "आंसू", "बेबस", "लाचार",
            "udas", "udasi", "rona", "dukh", "gham", "tanha", "dard", "mayus", "akela", "takleef", "aansu"
        ],
        "weight": 1.1
    },
    "Fear": {
        "words": [
            "डर", "खौफ", "भय", "बचाओ", "खतरा", "कांपना", "दहशत", "घबराहट", "मौत", "नुकसान",
            "चिंता", "सहम", "कांप", "डरपोक",
            "darr", "dar", "khauf", "bhay", "bachao", "khatra", "dahshat", "ghabrahat", "mout", "chinta"
        ],
        "weight": 1.15
    },
    "Surprise": {
        "words": [
            "क्या", "सच", "हैरान", "यकीन", "चौंकाने", "अचानक", "कैसे", "आश्चर्य", "अद्भुत", "गजब",
            "kya", "sach", "hairan", "yakeen", "achanak", "aashcharya", "gajab", "omg"
        ],
        "weight": 1.05
    },
    "Disgust": {
        "words": [
            "घिनौना", "छी", "गंदा", "बदबू", "थू", "घटिया", "बदसूरत", "बेहूदा", "घृणा",
            "ghinona", "chhee", "chi", "ganda", "badbu", "ghatiya", "badsurat", "behuda"
        ],
        "weight": 1.2
    },
    "Neutral": {
        "words": [
            "हां", "ठीक", "अच्छा", "सामान्य", "बात", "सुनो", "चलो", "शायद", "हाल", "पता", "जानकारी",
            "haan", "theek", "thik", "achha", "baat", "suno", "chalo", "shayad", "samanya"
        ],
        "weight": 0.8
    }
}

# English Emotional Lexicon
ENGLISH_EMOTION_LEXICON = {
    "Angry": {
        "words": [
            "angry", "furious", "mad", "rage", "irritated", "hate", "annoyed", "pissed", "shut up",
            "stupid", "idiot", "nonsense", "yell", "hostile", "outrage", "disgusted", "offensive"
        ],
        "weight": 1.2
    },
    "Happy": {
        "words": [
            "happy", "joy", "glad", "delighted", "excited", "love", "wonderful", "great", "awesome",
            "fantastic", "celebrate", "congratulations", "smile", "laugh", "cheerful", "blessed", "win"
        ],
        "weight": 1.1
    },
    "Sad": {
        "words": [
            "sad", "depressed", "unhappy", "cry", "crying", "tears", "grief", "heartbroken", "lonely",
            "miserable", "sorrow", "pain", "hopeless", "down", "gloomy", "lost", "hurt"
        ],
        "weight": 1.1
    },
    "Fear": {
        "words": [
            "afraid", "scared", "fear", "terrified", "panic", "horror", "anxious", "nervous", "danger",
            "threat", "shaking", "help", "alarmed", "frightened", "dread"
        ],
        "weight": 1.15
    },
    "Surprise": {
        "words": [
            "surprised", "shocked", "amazed", "astonished", "wow", "unbelievable", "unexpected",
            "sudden", "incredible", "gasp", "really", "whoa"
        ],
        "weight": 1.05
    },
    "Disgust": {
        "words": [
            "disgusting", "gross", "nasty", "revolting", "yuck", "sickening", "vile", "repulsive",
            "foul", "awful", "horrible", "crap"
        ],
        "weight": 1.2
    },
    "Neutral": {
        "words": [
            "ok", "okay", "fine", "normal", "yes", "no", "information", "meeting", "report", "regular",
            "standard", "maybe", "alright", "neutral", "talk", "hear"
        ],
        "weight": 0.8
    }
}

# Negation words that invert emotion
NEGATIONS = {
    "en": ["not", "never", "no", "hardly", "barely", "scarcely", "isn't", "aren't", "don't", "can't"],
    "hi": ["नहीं", "ना", "मत", "nahi", "nahin", "na", "mat"]
}


def detect_language(text: str) -> str:
    """
    Detects if the text is primarily Hindi (Devanagari / Hinglish) or English.
    Returns 'hi' or 'en'.
    """
    if not text or not text.strip():
        return "en"

    # Check for Devanagari Unicode block (U+0900 to U+097F)
    devanagari_count = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
    total_letters = sum(1 for ch in text if ch.isalpha())

    if total_letters > 0 and (devanagari_count / total_letters) > 0.15:
        return "hi"

    # Check Hinglish dictionary density
    tokens = [t.lower() for t in re.findall(r'\b\w+\b', text)]
    hi_matches = 0
    for emo_data in HINDI_EMOTION_LEXICON.values():
        for w in emo_data["words"]:
            if w in tokens:
                hi_matches += 1

    if hi_matches >= 2:
        return "hi"

    return "en"


def analyze_text_emotion(text: str, forced_lang: str = "auto") -> dict:
    """
    Analyzes semantic emotion from a spoken transcript or input sentence.
    
    Returns:
        dict with:
            - primary_emotion: dominant emotion from text
            - confidence: float score (0.0 to 1.0)
            - emotion_distribution: dict {emotion: probability}
            - detected_language: 'en' or 'hi'
            - detected_keywords: list of emotional cues found
    """
    if not text or not text.strip():
        # Default uniform neutral distribution when text is empty
        dist = {emo: 1.0 / len(EMOTIONS) for emo in EMOTIONS}
        return {
            "primary_emotion": "Neutral",
            "confidence": 0.35,
            "emotion_distribution": dist,
            "detected_language": "en",
            "detected_keywords": []
        }

    lang = forced_lang if forced_lang in ["en", "hi"] else detect_language(text)
    clean_text = text.lower().strip()
    words = re.findall(r'\b[\w\u0900-\u097F]+\b', clean_text)

    lexicon = HINDI_EMOTION_LEXICON if lang == "hi" else ENGLISH_EMOTION_LEXICON
    scores = {emo: 0.05 for emo in EMOTIONS}
    detected_keywords = []

    # Check for negations in word window
    active_negations = NEGATIONS.get(lang, []) + NEGATIONS.get("en", [])

    for i, word in enumerate(words):
        # Look behind for negation
        is_negated = False
        if i > 0 and words[i - 1] in active_negations:
            is_negated = True
        elif i > 1 and words[i - 2] in active_negations:
            is_negated = True

        for emo, data in lexicon.items():
            for target_word in data["words"]:
                if word == target_word or target_word in word:
                    weight = data["weight"]
                    if is_negated:
                        # Negation inverts emotion (e.g. "not happy" boosts Sad or Neutral)
                        if emo == "Happy":
                            scores["Sad"] += weight * 0.8
                            detected_keywords.append(f"negated-{word} -> Sad")
                        elif emo == "Angry":
                            scores["Neutral"] += weight * 0.8
                            detected_keywords.append(f"negated-{word} -> Neutral")
                        else:
                            scores["Neutral"] += weight * 0.5
                    else:
                        scores[emo] += weight
                        detected_keywords.append(word)

    # Exclamation mark boost for Surprise / Anger / Joy
    exclamation_count = text.count("!")
    if exclamation_count > 0:
        scores["Surprise"] += 0.3 * min(exclamation_count, 3)
        if scores["Angry"] > 0.1:
            scores["Angry"] += 0.4 * min(exclamation_count, 3)
        if scores["Happy"] > 0.1:
            scores["Happy"] += 0.4 * min(exclamation_count, 3)

    # Softmax normalization to turn raw scores into calibrated probabilities
    raw_vals = [scores[emo] for emo in EMOTIONS]
    max_val = max(raw_vals)
    exp_vals = [pow(2.71828, val - max_val) for val in raw_vals]
    sum_exp = sum(exp_vals)
    probabilities = {emo: round(exp_v / sum_exp, 4) for emo, exp_v in zip(EMOTIONS, exp_vals)}

    # Determine highest
    primary_emo = max(probabilities, key=probabilities.get)
    confidence = probabilities[primary_emo]

    return {
        "primary_emotion": primary_emo,
        "confidence": confidence,
        "emotion_distribution": probabilities,
        "detected_language": lang,
        "detected_keywords": list(set(detected_keywords))
    }
