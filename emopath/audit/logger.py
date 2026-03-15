import json
from pathlib import Path


class AuditLogger:

    def __init__(self, output_path):

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record):

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")