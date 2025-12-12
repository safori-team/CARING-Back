from math import exp, sqrt, log
from typing import Dict, Tuple, Optional

# Emotion anchors for Valence (V) and Arousal (A)
EMOTION_VA: Dict[str, Tuple[float, float]] = {
    "happy":    (+0.80, +0.60),
    "sad":      (-0.70, -0.40),
    "neutral":  ( 0.00,  0.00),
    "angry":    (-0.70, +0.80),
    "fear":     (-0.60, +0.70),
    "surprise": ( 0.00, +0.85),
}

# Intensity 판단 기준 (x1000 기준)
INTENSITY_THRESHOLDS = {
    "very_weak": 200,    # 0 ~ 200: 매우 약함 (거의 중립)
    "weak": 500,         # 200 ~ 500: 약함
    "moderate": 800,     # 500 ~ 800: 보통
    "strong": 1100,      # 800 ~ 1100: 강함
    "very_strong": 1414, # 1100 ~ 1414: 매우 강함 (최대값)
}


def magnitude_to_arousal(magnitude: float, k: float = 3.0) -> float:
    """Map text magnitude to arousal in [0,1] using A_text = 1 - exp(-magnitude/k)."""
    if magnitude <= 0:
        return 0.0
    try:
        return max(0.0, min(1.0, 1.0 - exp(-magnitude / k)))
    except Exception:
        return 0.0


def audio_probs_to_VA(audio_probs: Dict[str, float]) -> Tuple[float, float]:
    """Compute (V_audio, A_audio) as weighted sum of anchors by audio_probs."""
    v = 0.0
    a = 0.0
    for emo, p in audio_probs.items():
        anchor = EMOTION_VA.get(emo)
        if not anchor:
            continue
        try:
            p_val = float(p)
        except Exception:
            p_val = 0.0
        v += p_val * anchor[0]
        a += p_val * anchor[1]
    return v, a


def adaptive_weights_for_valence(audio_probs: Dict[str, float], score: float, a_text: float) -> float:
    """Compute alpha (text vs audio weight) for Valence.
    alpha = conf_audio_V / (conf_audio_V + conf_text_V + eps)
    where conf_text_V = |score|*A_text, conf_audio_V = max(p_e)
    """
    eps = 1e-6
    conf_text_v = abs(score) * max(0.0, min(1.0, a_text))
    conf_audio_v = max([0.0] + [float(p) for p in audio_probs.values()])
    denom = conf_audio_v + conf_text_v + eps
    return 0.0 if denom == 0 else max(0.0, min(1.0, conf_audio_v / denom))


def adaptive_weights_for_arousal(audio_probs: Dict[str, float], a_audio: float, a_text: float) -> float:
    """Compute beta for Arousal.
    beta = conf_audio_A / (conf_audio_A + conf_text_A + eps)
    where conf_text_A = A_text, conf_audio_A = |A_audio|
    """
    eps = 1e-6
    conf_text_a = max(0.0, min(1.0, a_text))
    conf_audio_a = abs(a_audio)
    denom = conf_audio_a + conf_text_a + eps
    return 0.0 if denom == 0 else max(0.0, min(1.0, conf_audio_a / denom))


def _cosine_similarity(x: Tuple[float, float], y: Tuple[float, float]) -> float:
    """Cosine similarity between two 2D vectors, clipped to [0,1] (negatives -> 0)."""
    vx, vy = x
    ax, ay = y
    num = vx * ax + vy * ay
    den = (sqrt(vx * vx + vy * vy) * sqrt(ax * ax + ay * ay)) or 1e-8
    sim = num / den
    return max(0.0, sim)


def _rbf_similarity(x: Tuple[float, float], y: Tuple[float, float], sigma: float = 0.75) -> float:
    """RBF kernel similarity between two 2D vectors (distance-based).
    
    Args:
        x: (V, A) tuple
        y: (V, A) tuple
        sigma: RBF kernel bandwidth (0.6~0.9 권장)
    
    Returns:
        Similarity score in [0, 1]
    """
    vx, ax = x
    vy, ay = y
    dist_sq = (vx - vy) ** 2 + (ax - ay) ** 2
    return exp(-(dist_sq) / (2 * sigma * sigma))


def _normalize_to_bps(scores: Dict[str, float]) -> Dict[str, int]:
    """Normalize non-negative scores to sum=10000 (bps), with rounding diff fix."""
    total = sum(max(0.0, s) for s in scores.values())
    if total <= 0:
        # fallback: neutral only
        base = {k: 0 for k in scores.keys()}
        if "neutral" in base:
            base["neutral"] = 10000
        else:
            # assign all to max key if exists
            if scores:
                first_key = next(iter(scores))
                base[first_key] = 10000
        return base
    scaled = {k: int(round(max(0.0, s) * (10000.0 / total))) for k, s in scores.items()}
    diff = 10000 - sum(scaled.values())
    if diff != 0 and scaled:
        key_max = max(scaled, key=lambda k: scaled[k])
        scaled[key_max] = max(0, min(10000, scaled[key_max] + diff))
    return scaled


def to_bps_from_unit_minus1_1(x: float) -> int:
    """Map [-1,1] -> [0,10000] linearly."""
    try:
        xv = max(-1.0, min(1.0, float(x)))
        return int(round((xv + 1.0) * 5000.0))
    except Exception:
        return 0


def to_x1000(x: float) -> int:
    """Scale float to int×1000 with rounding."""
    try:
        return int(round(float(x) * 1000.0))
    except Exception:
        return 0


def interpret_intensity(intensity_x1000: int) -> str:
    """intensity_x1000 값을 감정 강도 레벨로 해석.
    
    Args:
        intensity_x1000: intensity 값 (×1000 스케일)
        
    Returns:
        "very_weak", "weak", "moderate", "strong", "very_strong" 중 하나
    """
    if intensity_x1000 <= INTENSITY_THRESHOLDS["very_weak"]:
        return "very_weak"
    elif intensity_x1000 <= INTENSITY_THRESHOLDS["weak"]:
        return "weak"
    elif intensity_x1000 <= INTENSITY_THRESHOLDS["moderate"]:
        return "moderate"
    elif intensity_x1000 <= INTENSITY_THRESHOLDS["strong"]:
        return "strong"
    else:
        return "very_strong"


def get_intensity_level_kr(intensity_x1000: int) -> str:
    """intensity_x1000 값을 한국어 레벨로 반환.
    
    Returns:
        "매우 약함", "약함", "보통", "강함", "매우 강함" 중 하나
    """
    level_map = {
        "very_weak": "매우 약함",
        "weak": "약함",
        "moderate": "보통",
        "strong": "강함",
        "very_strong": "매우 강함",
    }
    level = interpret_intensity(intensity_x1000)
    return level_map.get(level, "알 수 없음")


def apply_zero_prob_mask(
    sims: Dict[str, float],
    audio_probs: Dict[str, float],
    *,
    threshold: float = 0.0,   # p ≤ threshold면 마스킹 (0.0이면 p==0만)
    mode: str = "soft",       # "hard": sims[e]=0, "soft": sims[e]*factor
    factor: float = 0.25      # 소프트 마스킹: 0으로 죽이지 말고 25%만 남김
) -> Dict[str, float]:
    out = dict(sims)
    for e, p in audio_probs.items():
        try:
            pv = float(p)
        except Exception:
            pv = 0.0
        if pv <= threshold and e in out:
            if mode == "hard":
                out[e] = 0.0
            else:
                # 소프트 마스킹: 0으로 죽이지 말고 factor만큼만 남김
                out[e] = max(0.0, out[e] * max(0.0, min(1.0, factor)))
    return out


def compute_entropy(probs: Dict[str, float]) -> float:
    """정규화된 엔트로피 계산 (0~1).
    
    감정 분포가 균일할수록(모든 감정이 비슷한 확률) 1에 가깝고,
    특정 감정에 집중될수록 0에 가깝습니다.
    
    Args:
        probs: 감정별 확률 딕셔너리 (합이 1일 필요 없음)
        
    Returns:
        정규화된 엔트로피 값 (0~1)
    """
    eps = 1e-10
    # 정규화
    total = sum(max(0.0, p) for p in probs.values())
    if total <= 0:
        return 1.0  # 모든 값이 0이면 최대 엔트로피(균일)로 간주
    
    normalized = {k: max(0.0, v) / total for k, v in probs.items()}
    
    # 엔트로피 계산
    h = -sum(p * log(p + eps) for p in normalized.values() if p > 0)
    max_h = log(len(probs)) if len(probs) > 0 else 1.0  # 균등 분포일 때 최대 엔트로피
    
    return h / max_h if max_h > 0 else 0.0


def fuse_VA(audio_probs: Dict[str, float], text_score: float, text_magnitude: float) -> Dict[str, object]:
    """Fuse audio (emotion probabilities) and text (score,magnitude) into composite VA.
    
    Returns dict with keys:
      - V_final, A_final, intensity, V_audio, A_audio, V_text, A_text, alpha, beta (float)
      - per_emotion_bps (dict[str,int], sum=10000), top_emotion (str), top_confidence_bps (int)
    """
    # Audio -> VA
    v_audio, a_audio = audio_probs_to_VA(audio_probs)

    # Text -> VA
    v_text = max(-1.0, min(1.0, float(text_score)))
    a_text = magnitude_to_arousal(float(text_magnitude))

    # Adaptive weights
    alpha = adaptive_weights_for_valence(audio_probs, v_text, a_text)
    beta = adaptive_weights_for_arousal(audio_probs, a_audio, a_text)

    # Final fusion (VA)
    v_final = alpha * v_audio + (1.0 - alpha) * v_text
    a_final = beta * a_audio + (1.0 - beta) * a_text
    intensity = sqrt(v_final * v_final + a_final * a_final)

    # Late Fusion: 감정별 확률 분포 결합 (neutral 과대 방지 분배)
    pos = max(0.0, v_text)
    neg = max(0.0, -v_text)
    mag = max(0.0, min(1.0, a_text))
    neutral_base = (1.0 - abs(v_text)) * (1.0 - mag)
    text_emotion_weight: Dict[str, float] = {
        "happy": pos * mag,
        "sad": neg * mag,
        "neutral": max(0.0, neutral_base),
        "angry": neg * mag,         # 부정 감정 동일 가중치
        "fear": neg * mag,          # 부정 감정 동일 가중치
        "surprise": pos * mag * 0.8,
    }
    # 긍정 텍스트( v_text > 0 )일 때 happy 동적 가중(증가) + surprise 경감, 이후 재정규화
    if v_text > 0:
        # 긍정일 때 happy 가중을 더 높임
        boost = 1.0 + 1.2 * mag * float(abs(v_text))   # 최대 +1.3배까지 추가 가중 (cap 아래에서 제한)
        if boost > 1.3:
            boost = 1.3
        damp  = max(0.5, 1.0 - 0.4 * mag * float(abs(v_text)))  # surprise는 최소 0.5배까지 감쇠
        text_emotion_weight["happy"] = text_emotion_weight.get("happy", 0.0) * boost
        text_emotion_weight["surprise"] = text_emotion_weight.get("surprise", 0.0) * damp
    # 재정규화
    t_sum = sum(text_emotion_weight.values())
    if t_sum > 0:
        for k in text_emotion_weight:
            text_emotion_weight[k] = text_emotion_weight[k] / t_sum

    # 감정별 분포 결합에서 텍스트 비중을 구간별로 가중
    # - |v_text| <= 0.5: 기존 가중 유지 (base)
    # - |v_text| > 0.5: 텍스트 비중 추가 상승 (strong)
    mag = max(0.0, min(1.0, a_text))
    abs_v = abs(float(v_text))
    base = 0.3 + 0.5 * mag * abs_v
    if abs_v <= 0.5:
        beta_prob = base
    else:
        # 임계 초과분 만큼 추가 가중 (최대 ~0.25), 상한 0.9
        extra = 0.5 * mag * (abs_v - 0.5)  # max 0.5*mag*0.5 = 0.25
        beta_prob = base + extra
    if beta_prob < 0.3:
        beta_prob = 0.3
    elif beta_prob > 0.9:
        beta_prob = 0.9
    alpha_prob = 1.0 - beta_prob
    composite_score: Dict[str, float] = {}
    for emo in ["happy", "sad", "neutral", "angry", "fear", "surprise"]:
        a_sc = float(audio_probs.get(emo, 0.0))
        t_sc = float(text_emotion_weight.get(emo, 0.0))
        composite_score[emo] = alpha_prob * a_sc + beta_prob * t_sc

    # Audio에서 분노 비율이 매우 낮은 경우(angry_bps <= 2000 → angry_prob <= 0.2),
    # voice_composite에서 분노가 과도하게 top으로 나오는 것을 방지하기 위해
    # audio angry 정보의 최종 기여도를 완만하게 줄인다.
    try:
        angry_p = float(audio_probs.get("angry", 0.0))
        if angry_p <= 0.2:
            neg = max(0.0, -float(v_text))
            mag = max(0.0, min(1.0, float(a_text)))
            base_factor = 0.7
            extra_down = 0.15 * neg * mag   # 최대 약 0.15 추가 감쇠
            factor = max(0.5, base_factor - extra_down)  # 최소 0.5배까지
            composite_score["angry"] = composite_score.get("angry", 0.0) * factor
    except Exception:
        # 로직 실패 시에는 안전하게 무시
        pass

    # 감정별 가중치 조정: neutral은 더 강하게 억제(긍정일수록 추가 억제)
    neutral_base_factor = 0.6
    if v_text > 0:
        # v_text, a_text가 클수록 neutral 추가 감쇠 (최소 0.3배까지)
        extra_down = 0.2 * max(0.0, min(1.0, a_text)) * float(abs(v_text))
        neutral_factor = max(0.3, neutral_base_factor - extra_down)
    else:
        neutral_factor = neutral_base_factor
    
    # 충돌 감지: v_audio와 v_text 부호가 다르면 감정 상쇄 발생
    # 이 경우 neutral이 과대 평가되므로 추가 억제
    is_conflict = (v_audio * v_text) < 0
    if is_conflict:
        conflict_factor = 0.1  # 충돌 시 neutral 0.1배로 강하게 억제
    else:
        conflict_factor = 1.0
    
    # 엔트로피 기반 억제: 감정 분포가 균일할수록(엔트로피 높음) neutral 추가 억제
    entropy = compute_entropy(composite_score)
    if entropy > 0.8:
        entropy_factor = 0.3  # 높은 엔트로피 시 0.3배
    elif entropy > 0.6:
        entropy_factor = 0.6  # 중간 엔트로피 시 0.6배
    else:
        entropy_factor = 1.0
    
    # 최종 neutral 억제: 기존 + 충돌 + 엔트로피
    composite_score["neutral"] = composite_score.get("neutral", 0.0) * neutral_factor * 0.7 * conflict_factor * entropy_factor
    composite_score["surprise"] = composite_score.get("surprise", 0.0) * 0.9

    per_emotion_bps = _normalize_to_bps(composite_score)

    # Top emotion/confidence
    if per_emotion_bps:
        top_emotion = max(per_emotion_bps, key=lambda k: per_emotion_bps[k])
        top_confidence_bps = per_emotion_bps[top_emotion]
    else:
        top_emotion = "neutral"
        top_confidence_bps = 10000

    return {
        "V_final": v_final,
        "A_final": a_final,
        "intensity": intensity,
        "V_audio": v_audio,
        "A_audio": a_audio,
        "V_text": v_text,
        "A_text": a_text,
        "alpha": alpha,
        "beta": beta,
        "per_emotion_bps": per_emotion_bps,
        "top_emotion": top_emotion,
        "top_confidence_bps": top_confidence_bps,
    }