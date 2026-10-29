"""
Multimodal Fusion Module
Combines acoustic speech emotion probabilities with NLP transcript probabilities
to produce a robust cross-modal emotion prediction.
"""

from src.config import EMOTIONS

def fuse_multimodal_predictions(
    acoustic_probs: dict[str, float],
    nlp_probs: dict[str, float] | None = None,
    audio_weight: float = 0.70
) -> dict:
    """
    Fuses acoustic features and linguistic NLP text emotion probabilities.
    
    Args:
        acoustic_probs: dict {emotion: probability} from acoustic model
        nlp_probs: dict {emotion: probability} from NLP text analysis (optional)
        audio_weight: float between 0.0 and 1.0 (default 0.70 for 70% audio, 30% text)
        
    Returns:
        dict with:
            - fused_emotion: str
            - confidence: float (0.0 to 1.0)
            - confidence_percentage: float (0 - 100%)
            - distribution: dict {emotion: score}
            - modal_agreement: bool (whether audio and text predicted the same emotion)
            - fusion_type: "bimodal" or "acoustic_only"
    """
    if not nlp_probs or audio_weight >= 1.0:
        # Acoustic only
        best_emo = max(acoustic_probs, key=acoustic_probs.get)
        conf = acoustic_probs[best_emo]
        return {
            "fused_emotion": best_emo,
            "confidence": round(conf, 4),
            "confidence_percentage": round(conf * 100, 1),
            "distribution": acoustic_probs,
            "modal_agreement": True,
            "fusion_type": "acoustic_only"
        }

    text_weight = 1.0 - audio_weight
    fused_scores = {}

    for emo in EMOTIONS:
        a_score = acoustic_probs.get(emo, 0.0)
        t_score = nlp_probs.get(emo, 0.0)
        fused_scores[emo] = (audio_weight * a_score) + (text_weight * t_score)

    # Normalize so sum equals 1.0
    total = sum(fused_scores.values()) or 1.0
    fused_dist = {emo: round(score / total, 4) for emo, score in fused_scores.items()}

    best_emo = max(fused_dist, key=fused_dist.get)
    best_conf = fused_dist[best_emo]

    acoustic_top = max(acoustic_probs, key=acoustic_probs.get)
    nlp_top = max(nlp_probs, key=nlp_probs.get)
    modal_agreement = (acoustic_top == nlp_top)

    return {
        "fused_emotion": best_emo,
        "confidence": best_conf,
        "confidence_percentage": round(best_conf * 100, 1),
        "distribution": fused_dist,
        "modal_agreement": modal_agreement,
        "acoustic_top": acoustic_top,
        "nlp_top": nlp_top,
        "fusion_type": "bimodal"
    }
