import re


def find_matching_words_in_text(text, words):
    text = text.lower()

    ws = [word.lower().strip() for word in words if word and str(word).strip()]
    ws.sort(key=lambda word: (-len(word.split()), -len(word)))

    matching_words = []
    matched_spans = []

    for word in ws:
        # 將 keyword 做 regex escape，避免特殊字元影響
        escaped_word = re.escape(word)

        # 無論單字或片語，都用 word boundary 方式處理
        # 例如 "book" 不會匹配 "booked"
        pattern = rf"\b{escaped_word}\b"

        for match in re.finditer(pattern, text):
            span = match.span()

            # 避免較短詞重複吃到已被較長詞覆蓋的區段
            if any(not (span[1] <= s[0] or span[0] >= s[1]) for s in matched_spans):
                continue

            matching_words.append(word)
            matched_spans.append(span)
            break

    count = len(matching_words)

    return count, matching_words


def process_emotions(text, dictionary):
    emotion_results = {}

    for emotion, keywords in dictionary.items():
        count, matching_words = find_matching_words_in_text(text, keywords)

        emotion_results[emotion] = {
            "count": count,
            "matching_words": matching_words
        }

    return emotion_results