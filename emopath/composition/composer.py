def compose_emotions(emotion_scores, threshold=0.5):

    dominant = max(emotion_scores, key=emotion_scores.get)

    secondary = [
        e for e,score in emotion_scores.items()
        if score >= threshold and e != dominant
    ]

    return {
        "dominant_emotion": dominant,
        "secondary_emotions": secondary
    }