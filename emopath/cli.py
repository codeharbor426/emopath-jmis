import argparse
import json
from datetime import datetime

from emopath.pipeline import (
    run_pipeline,
)
from emopath.generation.prompt_builder import PromptBuilder
from emopath.generation.generator import ResponseGenerator


APPRAISAL_MAP = {
    "1_Anger": "perceived injustice and external blame",
    "2_Frustration": "goal blockage caused by uncontrollable obstacles",
    "3_Disappointment": "unmet expectations and expectancy disconfirmation",
    "4_Helplessness": "low coping potential and loss of control",
    "5_Anxiety": "future threat and uncertainty with low control"
}

DISPLAY_EMOTION_MAP = {
    "1_Anger": "Anger",
    "2_Frustration": "Frustration",
    "3_Disappointment": "Disappointment",
    "4_Helplessness": "Helplessness",
    "5_Anxiety": "Anxiety"
}


def run_cli_pipeline(text, condition="C1", threshold=0.5, show_prompt=False):
    """
    CLI 單筆推理模式：
    1. 跑 EmoPath detection / composition / strategy
    2. 依 condition 組 prompt
    3. 呼叫 GPT 生成回覆
    4. 回傳完整 JSON
    """

    # -------------------------
    # Step 1: base EmoPath pipeline
    # -------------------------
    pipeline_result = run_pipeline(text, threshold)

    # -------------------------
    # Step 2: extract variables
    # -------------------------
    emotion_vector = pipeline_result["emotion_scores"]
    dominant = pipeline_result["dominant_emotion"]
    secondary = pipeline_result["secondary_emotions"]
    strategy_plan = pipeline_result["strategy_plan"]
    emotion_markers = pipeline_result.get("emotion_markers", {})

    # C2: No composition
    if condition == "C2":
        secondary = []
        pipeline_result["secondary_emotions"]

    # -------------------------
    # Step 3: rationale
    # -------------------------
    dominant_display = DISPLAY_EMOTION_MAP.get(dominant, dominant)
    rationale = (
        f"{dominant_display} dominant due to "
        f"{APPRAISAL_MAP.get(dominant, 'customer appraisal pattern')}"
    )

    # -------------------------
    # Step 4: build prompt
    # CLI mode 不做 policy 檢查，所以 policy_document 留空字串
    # -------------------------
    builder = PromptBuilder()

    complaint_data = {
        "text": text,
        "policy_document": ""
    }

    bundle = builder.build(
        complaint_data=complaint_data,
        pipeline_result=pipeline_result,
        condition=condition
    )

    # -------------------------
    # Step 5: generate response
    # -------------------------
    generator = ResponseGenerator()

    gen = generator.generate(
        system_prompt=bundle["system"],
        user_prompt=bundle["user"]
    )

    response_text = gen["response"]

    result = {
        "response": response_text,
        "emotion_vector": emotion_vector,
        "dominant_emotion": dominant,
        "secondary_emotions": secondary,
        "strategy_plan": strategy_plan,
        "model_version": gen["model_version"],
        "random_seed": gen["random_seed"],
        "emotion_markers": emotion_markers,
        "rationale": rationale,
        "timestamp": datetime.utcnow().isoformat()
    }

    if show_prompt:
        result["prompt"] = {
            "system": bundle["system"],
            "user": bundle["user"]
        }

    return result


def main():

    parser = argparse.ArgumentParser(description="EmoPath CLI")

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Customer complaint text"
    )

    parser.add_argument(
        "--condition",
        type=str,
        default="C1",
        choices=["C1", "C2", "C3", "C4"],
        help="Prompt condition (default: C1)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Secondary emotion threshold (default: 0.5)"
    )

    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Show system/user prompts in output"
    )

    args = parser.parse_args()

    result = run_cli_pipeline(
        text=args.text,
        condition=args.condition,
        threshold=args.threshold,
        show_prompt=args.show_prompt
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()