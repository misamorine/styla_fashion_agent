import torch
from transformers import CLIPModel, CLIPProcessor

class FashionEmbedder:
    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def text_embedding(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embedding = embedding = self.model.get_text_features(**inputs)
            embedding = embedding.pooler_output  # Extract the tensor
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0].tolist()

    def image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs) 
            embedding = embedding.pooler_output  # Extract the tensor
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0].tolist()
