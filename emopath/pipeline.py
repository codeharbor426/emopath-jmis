import json
import argparse
from datetime import datetime

from emopath.detection.dictionary_loader import load_dictionary
from emopath.detection.detector import process_emotions
from emopath.composition.composer import compose_emotions
from emopath.planner.strategy import extract_emotion_scores, build_strategy
from emopath.generation.prompt_builder import PromptBuilder
from emopath.generation.generator import ResponseGenerator
from emopath.generation.pvi_checker import PVIChecker
from emopath.audit.logger import AuditLogger

APPRAISAL_MAP = {
    "1_Anger": "perceived injustice and external blame",
    "2_Frustration": "goal blockage caused by uncontrollable obstacles",
    "3_Disappointment": "unmet expectations and expectancy disconfirmation",
    "4_Helplessness": "low coping potential and loss of control",
    "5_Anxiety": "future threat and uncertainty with low control"
}

def extract_emotion_markers(emotion_results):
    """
    Extract matched lexicon words for each emotion
    """

    markers = {}

    for emotion, result in emotion_results.items():

        words = result.get("matching_words", [])

        if words:
            markers[emotion] = words

    return markers

def run_pipeline(text, threshold=0.5):
    """
    主 pipeline：串接 EmoPath 各模組
    """

    # Step 1 載入情緒字典
    dictionary = load_dictionary()

    # Step 2 情緒偵測
    emotion_results = process_emotions(text.lower(), dictionary)

    # Step 3 取得 emotion scores
    emotion_scores = extract_emotion_scores(emotion_results)
    emotion_markers = extract_emotion_markers(emotion_results)

    # Step 4 Emotion composition
    composition = compose_emotions(emotion_scores, threshold)

    dominant_emotion = composition["dominant_emotion"]
    secondary_emotions = composition["secondary_emotions"]

    # Step 5 建立策略
    strategy_plan = build_strategy(dominant_emotion)

    result = {
        "emotion_scores": emotion_scores,
        "emotion_markers": emotion_markers,
        "dominant_emotion": dominant_emotion,
        "secondary_emotions": secondary_emotions,
        "strategy_plan": strategy_plan
    }

    return result

def load_jsonl(path):

    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def build_policy_lookup(policy_path):

    policies = load_jsonl(policy_path)

    lookup = {}

    for p in policies:

        lookup[p["policy_doc_id"]] = p

    return lookup


def run_ablation_pipeline(args):

    inputs = load_jsonl(args.input)

    policy_lookup = build_policy_lookup(args.policy_doc)

    builder = PromptBuilder()

    generator = ResponseGenerator()

    pvi = PVIChecker()

    logger = AuditLogger(
        f"{args.out}/ablation_outputs.jsonl"
    )

    for row in inputs:

        input_id = row["input_id"]

        complaint_text = row["text"]

        policy = policy_lookup[input_id]

        policy_text = policy["policy_text"]

        row["policy_document"] = policy_text

        # -------------------------
        # run EmoPath pipeline
        # -------------------------

        pipeline_result = run_pipeline(complaint_text)

        # -------------------------
        # build prompt
        # -------------------------

        bundle = builder.build(
            complaint_data=row,
            pipeline_result=pipeline_result,
            condition=args.condition
        )

        # -------------------------
        # generate response
        # -------------------------

        if args.condition == "C5":

            response_text = bundle["template_response"]

            model_version = "template"

            random_seed = 42

        else:

            gen = generator.generate(
                system_prompt=bundle["system"],
                user_prompt=bundle["user"]
            )

            response_text = gen["response"]

            model_version = gen["model_version"]

            random_seed = gen["random_seed"]

        # -------------------------
        # guardrail
        # -------------------------

        guard = pvi.check(
            response_text=response_text,
            complaint_text=complaint_text,
            policy=policy
        )

        # -------------------------
        # output record
        # -------------------------
        dominant = pipeline_result["dominant_emotion"]
        rationale = f"{dominant} dominant due to {APPRAISAL_MAP.get(dominant, 'customer appraisal pattern')}"

        record = {

            "input_id": input_id,

            "condition": args.condition,

            "response": response_text,

            "emotion_vector": pipeline_result["emotion_scores"],

            "dominant_emotion": pipeline_result["dominant_emotion"],

            "secondary_emotions": bundle["secondary_emotions"],

            "strategy_plan": bundle["strategy_plan"],

            "guardrail_triggered": guard["guardrail_triggered"],

            "risk_flags": guard["risk_flags"],

            "model_version": model_version,

            "random_seed": random_seed,

            "emotion_markers": pipeline_result.get("emotion_markers", {}),

            "guardrail_results": guard["guardrail_results"],

            "rationale": rationale,

            "timestamp": datetime.utcnow().isoformat()
        }

        logger.append(record)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)

    parser.add_argument("--policy_doc", required=True)

    parser.add_argument("--out", required=True)

    parser.add_argument("--condition", required=True)

    args = parser.parse_args()

    run_ablation_pipeline(args)


if __name__ == "__main__":
    main()