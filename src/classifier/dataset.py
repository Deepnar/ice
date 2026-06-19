import json
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class ICEClassifierDataset(Dataset):
    def __init__(self, training_data_path):
        self.data = []
        prompts = []
        labels_list = []

        with open(training_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append(item)
                prompts.append(item["prompt"])
                labels_list.append(item["labels"])

        # Convert labels to tensor once
        self.labels = torch.tensor(labels_list, dtype=torch.float32)

        # Pre-compute embeddings with the frozen sentence transformer
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu",truncate_dim=384)
        self.embeddings = model.encode(prompts, convert_to_tensor=True, show_progress_bar=True).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]