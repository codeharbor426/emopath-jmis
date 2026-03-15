import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer


class EmotionClassifier(nn.Module):

    def __init__(self, model_name="bert-base-uncased", num_labels=5):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls)

        return logits