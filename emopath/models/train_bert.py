import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW

from .bert_classifier import EmotionClassifier
from .dataset import EmotionDataset


EMOTIONS = [
    "anger",
    "frustration",
    "disappointment",
    "helplessness",
    "anxiety"
]


def train_model(train_texts, train_labels, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = EmotionDataset(
        train_texts,
        train_labels,
        tokenizer
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = EmotionClassifier(model_name)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(3):

        for batch in loader:

            optimizer.zero_grad()

            logits = model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            loss = loss_fn(
                logits,
                batch["labels"]
            )

            loss.backward()

            optimizer.step()

    return model, tokenizer

def predict(model, tokenizer, texts):

    model.eval()

    dataset = EmotionDataset(
        texts,
        [[0]*5]*len(texts),
        tokenizer
    )

    loader = DataLoader(dataset, batch_size=8)

    preds = []

    with torch.no_grad():

        for batch in loader:

            logits = model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            probs = torch.sigmoid(logits)

            preds.append(probs.cpu())

    preds = torch.cat(preds)

    return (preds > 0.5).int().numpy()