
def extract_emotion_scores(emotion_results):
    """
    將 detector 的結果轉為 emotion_scores
    """
    return {emotion: result['count'] for emotion, result in emotion_results.items()}


def build_strategy(dominant_emotion):
    """
    根據 dominant emotion 建立 recovery strategy
    """

    if dominant_emotion == '1_Anger':
        return {
            "stage1": "active listening and acknowledge injustice",
            "stage2": "offer compensation within policy limits"
        }

    elif dominant_emotion == '2_Frustration':
        return {
            "stage1": "express empathy for the inconvenience",
            "stage2": "provide retrospective explanation of the issue"
        }

    elif dominant_emotion == '3_Disappointment':
        return {
            "stage1": "acknowledge unmet expectations",
            "stage2": "provide retrospective and prospective explanation"
        }

    elif dominant_emotion == '4_Helplessness':
        return {
            "stage1": "show empathy and reassurance",
            "stage2": "provide clear future resolution steps"
        }

    elif dominant_emotion == '5_Anxiety':
        return {
            "stage1": "acknowledge customer concerns",
            "stage2": "provide risk-reducing information"
        }

    else:
        return {
            "stage1": "acknowledge customer concern",
            "stage2": "provide assistance within policy"
        }