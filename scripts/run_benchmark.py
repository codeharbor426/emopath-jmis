import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, hamming_loss, accuracy_score
from emopath.models.train_bert import train_model, predict
from emopath.pipeline import run_pipeline
from dotenv import load_dotenv
import os
from openai import OpenAI
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

EMOTIONS = [
    "anger",
    "frustration",
    "disappointment",
    "helplessness",
    "anxiety"
]


EMOTION_MAP = {
    "1_Anger": "anger",
    "2_Frustration": "frustration",
    "3_Disappointment": "disappointment",
    "4_Helplessness": "helplessness",
    "5_Anxiety": "anxiety"
}

def load_prompt_config(path):

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

def run_gpt4(inputs, test_ids, prompt_config, mode="zero"):

    predictions = []

    system_prompt = prompt_config["system_message"]

    if mode == "zero":
        template = prompt_config["zero_shot_user_template"]
    else:
        template = prompt_config["few_shot_user_template"]

    model = prompt_config["model"]

    for item in inputs:

        sample_id = item["input_id"]

        if sample_id not in test_ids:
            continue

        text = item["text"]

        user_prompt = template.replace("{text}", text)

        response = client.chat.completions.create(
            model=model,
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        output = response.choices[0].message.content.strip()

        row = {
            "id": sample_id,
            "anger": 0,
            "frustration": 0,
            "disappointment": 0,
            "helplessness": 0,
            "anxiety": 0
        }

        try:
            # 嘗試抽取 JSON
            json_match = re.search(r"\{.*\}", output, re.DOTALL)

            if json_match:
                json_str = json_match.group()
                pred = json.loads(json_str)

                for emotion in EMOTIONS:
                    if emotion in pred:
                        row[emotion] = int(pred[emotion])

            else:
                print("JSON not found:", output)

        except Exception as e:
            print("JSON parse failed:", output)

        predictions.append(row)

    return pd.DataFrame(predictions)


def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def load_test_ids(folds_path):

    with open(folds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return set(data["test_ids"])


def run_lexicon(inputs, test_ids):

    predictions = []

    for item in inputs:

        sample_id = item["input_id"]

        if sample_id not in test_ids:
            continue

        text = item["text"]

        result = run_pipeline(text)

        dominant = result.get("dominant_emotion")
        secondary = result.get("secondary_emotion")

        label1 = EMOTION_MAP.get(dominant)
        label2 = EMOTION_MAP.get(secondary)

        row = {
            "id": sample_id,
            "anger": 0,
            "frustration": 0,
            "disappointment": 0,
            "helplessness": 0,
            "anxiety": 0
        }

        if label1:
            row[label1] = 1

        if label2:
            row[label2] = 1

        predictions.append(row)

    return pd.DataFrame(predictions)


def evaluate(pred_df, gold_df):

    gold_df = gold_df[["id"] + EMOTIONS]

    merged = pred_df.merge(gold_df, on="id", suffixes=("_pred", "_true"))

    y_true = merged[[f"{e}_true" for e in EMOTIONS]].values
    y_pred = merged[[f"{e}_pred" for e in EMOTIONS]].values

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    h_loss = hamming_loss(y_true, y_pred)

    subset_acc = accuracy_score(
        list(map(tuple, y_true)),
        list(map(tuple, y_pred))
    )

    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None
    )

    per_label = {
        emotion: {
            "precision": pr[i],
            "recall": rc[i],
            "f1": f1[i]
        }
        for i, emotion in enumerate(EMOTIONS)
    }

    metrics = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": h_loss,
        "subset_accuracy": subset_acc
    }

    return merged, metrics, per_label

def run_bert(inputs, gold_df, test_ids):

    train_df = gold_df[~gold_df["id"].isin(test_ids)]

    texts = []
    labels = []

    for _, row in train_df.iterrows():

        text = next(x["text"] for x in inputs if x["input_id"] == row["id"])

        texts.append(text)

        labels.append([
            row["anger"],
            row["frustration"],
            row["disappointment"],
            row["helplessness"],
            row["anxiety"]
        ])

    model, tokenizer = train_model(
        texts,
        labels,
        "bert-base-uncased"
    )

    test_texts = [
        x["text"]
        for x in inputs
        if x["input_id"] in test_ids
    ]

    preds = predict(
        model,
        tokenizer,
        test_texts
    )

    ids = [
        x["input_id"]
        for x in inputs
        if x["input_id"] in test_ids
    ]

    rows = []

    for i, pid in enumerate(ids):

        rows.append({
            "id": pid,
            "anger": preds[i][0],
            "frustration": preds[i][1],
            "disappointment": preds[i][2],
            "helplessness": preds[i][3],
            "anxiety": preds[i][4],
        })

    return pd.DataFrame(rows)

def run_roberta(inputs, gold_df, test_ids):

    train_df = gold_df[~gold_df["id"].isin(test_ids)]

    texts = []
    labels = []

    for _, row in train_df.iterrows():

        text = next(x["text"] for x in inputs if x["input_id"] == row["id"])

        texts.append(text)

        labels.append([
            row["anger"],
            row["frustration"],
            row["disappointment"],
            row["helplessness"],
            row["anxiety"]
        ])

    model, tokenizer = train_model(
        texts,
        labels,
        "roberta-base"
    )

    test_texts = [
        x["text"]
        for x in inputs
        if x["input_id"] in test_ids
    ]

    preds = predict(
        model,
        tokenizer,
        test_texts
    )

    ids = [
        x["input_id"]
        for x in inputs
        if x["input_id"] in test_ids
    ]

    rows = []

    for i, pid in enumerate(ids):

        rows.append({
            "id": pid,
            "anger": preds[i][0],
            "frustration": preds[i][1],
            "disappointment": preds[i][2],
            "helplessness": preds[i][3],
            "anxiety": preds[i][4],
        })

    return pd.DataFrame(rows)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    data_path = Path(args.data)
    labels_path = Path(args.labels)
    folds_path = Path(args.folds)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    inputs = load_jsonl(data_path)

    print("Loading gold labels...")
    gold_df = pd.read_csv(labels_path)

    print("Loading test ids...")
    test_ids = load_test_ids(folds_path)

    print("Loading GPT prompt config...")
    prompt_config = load_prompt_config(
        "data/samples/gpt4_prompts_study3a.json"
    )

    results = []

    # NEW: 存每個 emotion 的 metrics
    per_label_results = []

    # -----------------------------
    # B1 EmoPath Lexicon
    # -----------------------------
    if "lexicon" in args.models:

        print("Running EmoPath Lexicon...")

        pred_df = run_lexicon(inputs, test_ids)

        merged, metrics, per_label = evaluate(pred_df, gold_df)

        pred_file = out_dir / "predictions_B1.csv"
        merged.to_csv(pred_file, index=False)

        results.append({
            "model": "EmoPath_Lexicon",
            **metrics
        })

        for emotion, scores in per_label.items():

            per_label_results.append({
                "model": "EmoPath_Lexicon",
                "emotion": emotion,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"]
            })

    # -----------------------------
    # B2 BERT multi-label
    # -----------------------------
    if "bert" in args.models:

        print("Running BERT multi-label...")

        pred_df = run_bert(
            inputs,
            gold_df,
            test_ids
        )

        merged, metrics, per_label = evaluate(pred_df, gold_df)

        pred_file = out_dir / "predictions_B2.csv"
        merged.to_csv(pred_file, index=False)

        results.append({
            "model": "BERT",
            **metrics
        })

        for emotion, scores in per_label.items():

            per_label_results.append({
                "model": "BERT",
                "emotion": emotion,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"]
            })

    # -----------------------------
    # B3 RoBERTa multi-label
    # -----------------------------
    if "roberta" in args.models:

        print("Running RoBERTa multi-label...")

        pred_df = run_roberta(
            inputs,
            gold_df,
            test_ids
        )

        merged, metrics, per_label = evaluate(pred_df, gold_df)

        pred_file = out_dir / "predictions_B3.csv"
        merged.to_csv(pred_file, index=False)

        results.append({
            "model": "RoBERTa",
            **metrics
        })

        for emotion, scores in per_label.items():

            per_label_results.append({
                "model": "RoBERTa",
                "emotion": emotion,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"]
            })

    # -----------------------------
    # B4 GPT4 Few-shot
    # -----------------------------
    if "gpt4_few" in args.models:

        print("Running GPT-4 few-shot...")

        pred_df = run_gpt4(inputs, test_ids, prompt_config, mode="few")

        merged, metrics, per_label = evaluate(pred_df, gold_df)

        pred_file = out_dir / "predictions_B4.csv"
        merged.to_csv(pred_file, index=False)

        results.append({
            "model": "GPT4_few",
            **metrics
        })

        for emotion, scores in per_label.items():

            per_label_results.append({
                "model": "GPT4_few",
                "emotion": emotion,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"]
            })

    # -----------------------------
    # B5 GPT4 Zero-shot
    # -----------------------------
    if "gpt4_zero" in args.models:

        print("Running GPT-4 zero-shot...")

        pred_df = run_gpt4(inputs, test_ids, prompt_config, mode="zero")

        merged, metrics, per_label = evaluate(pred_df, gold_df)

        pred_file = out_dir / "predictions_B5.csv"
        merged.to_csv(pred_file, index=False)

        results.append({
            "model": "GPT4_zero",
            **metrics
        })

        for emotion, scores in per_label.items():

            per_label_results.append({
                "model": "GPT4_zero",
                "emotion": emotion,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"]
            })

    # -----------------------------
    # Save benchmark results
    # -----------------------------
    results_df = pd.DataFrame(results)

    results_file = out_dir / "benchmark_results.csv"
    results_df.to_csv(results_file, index=False)

    print("Benchmark finished.")
    print(results_df)

    # NEW: per-label metrics
    per_label_df = pd.DataFrame(per_label_results)

    per_label_file = out_dir / "per_label_metrics.csv"
    per_label_df.to_csv(per_label_file, index=False)

    print("Per Label Metrics finished.")
    print(per_label_df)


if __name__ == "__main__":
    main()