import json
import re
from pathlib import Path


class PromptBuilder:

    def __init__(self, template_path="data/samples/gpt4_prompts_3b.json"):
        template_path = Path(template_path)

        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.meta = data["meta"]

        self.templates = {
            "C1": data["C1_full_emopath"],
            "C2": data["C2_no_composition"],
            "C3": data["C3_no_sequential"],
            "C4": data["C4_unconstrained"],
            "C5": data["C5_template"],
        }

    def build(self, complaint_data, pipeline_result, condition):

        if condition == "C5":
            return self._build_template_response(complaint_data)

        template = self.templates[condition]

        emotion_scores = pipeline_result["emotion_scores"]
        dominant = pipeline_result["dominant_emotion"]
        secondary = pipeline_result["secondary_emotions"]
        strategy = pipeline_result["strategy_plan"]

        stage1 = None
        stage2 = None

        if strategy != "n/a":
            stage1 = strategy.get("stage1")
            stage2 = strategy.get("stage2")

        if condition == "C2":
            secondary = []

        if condition == "C3":
            stage1 = None

        if condition in ["C4"]:
            strategy = "n/a"

        variables = {
            "policy_doc": complaint_data["policy_document"],
            "complaint": complaint_data["text"],
            "emotion_vector": emotion_scores,
            "dominant_emotion": dominant,
            "secondary_emotions": secondary,
            "stage1": stage1,
            "stage2": stage2,
        }

        user_prompt = template["user_template"].format(**variables)

        return {
            "system": template["system"],
            "user": user_prompt,
            "secondary_emotions": secondary,
            "strategy_plan": strategy
        }

    def _build_template_response(self, complaint_data):

        template = self.templates["C5"]["template_response"]

        policy = complaint_data["policy_document"]

        company_name = self._extract_company(policy)
        max_comp = self._extract_comp(policy)

        response = template.format(
            company_name=company_name,
            max_compensation=max_comp
        )

        return {
            "template_response": response,
            "secondary_emotions": [],
            "strategy_plan": "n/a"
        }

    def _extract_company(self, policy_doc):

        match = re.search(r"COMPANY:\s*(.+)", policy_doc)

        if match:
            return match.group(1).strip()

        return "Customer Service"

    def _extract_comp(self, policy_doc):

        match = re.search(r"MAX COMPENSATION:\s*(.+)", policy_doc)

        if match:
            return match.group(1).strip()

        return "an amount within policy limits"