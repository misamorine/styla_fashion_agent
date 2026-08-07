import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS pooling & normalization (standard BGE embedding format)
            embedding = outputs[0][:, 0]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()
