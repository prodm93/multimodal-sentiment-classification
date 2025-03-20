import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# class for preparing dataset for multimodal model
class MultiModalDataset(Dataset):
    def __init__(self, texts, image_dataset, labels, tokenizer, feature_extractor, max_length=128):
        self.texts = texts
        # Change: Instead of image paths, we now expect a HuggingFace dataset.
        # We assume the dataset contains a split named "train". Adjust if needed.
        self.image_dataset = image_dataset["train"]
        self.labels = labels
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.image_dataset[idx]["image"]
        label = self.labels[idx]

        # Tokenise text
        text_inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Remove batch dimension from text inputs
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        # Extract features from images
        image_inputs = self.feature_extractor(images=image, return_tensors='pt')
        image_inputs = {k: v.squeeze(0) for k, v in image_inputs.items()}

        return text_inputs, image_inputs, torch.tensor(label, dtype=torch.long) 
        # make sure inputs are returned in torch format compatible with fusion setup


# class for fusion of text and image models

class FusionClassifier(nn.Module):
    def __init__(self, text_model, image_model, num_classes, fusion_dropout=0.1):
        super(FusionClassifier, self).__init__()
        self.text_model = text_model
        self.image_model = image_model

        # Get text hidden size
        text_hidden_size = getattr(text_model.config, "hidden_size", 768)
        
        # Handle getting the hidden size of image model correctly
        # based on attribute name (this was causing issues at first)
        if hasattr(image_model.config, "hidden_size"):
            image_hidden_size = image_model.config.hidden_size
        elif hasattr(image_model.config, "embed_dim"):
            image_hidden_size = image_model.config.embed_dim
        elif hasattr(image_model.config, "hidden_sizes"):
            image_hidden_size = image_model.config.hidden_sizes[-1]
        else:
            image_hidden_size = 768  # default value consistent with common hidden size value

        print("Text hidden size:", text_hidden_size)
        print("Image hidden size:", image_hidden_size)

        fusion_dim = text_hidden_size + image_hidden_size
        self.dropout = nn.Dropout(fusion_dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, text_inputs, image_inputs):
        # Extract features from text
        text_outputs = self.text_model(**text_inputs)
        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_feat = text_outputs.pooler_output  
        else:
            text_feat = text_outputs.last_hidden_state.mean(dim=1)  

        # Extract features from image
        image_outputs = self.image_model(**image_inputs)
        if hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
            image_feat = image_outputs.pooler_output  
        else:
            # Flatten the image outputs based on existing dim to 2D so that the final dim matches text outputs
            if image_outputs.last_hidden_state.dim() == 4:
                image_feat = torch.nn.functional.adaptive_avg_pool2d(image_outputs.last_hidden_state, (1, 1))
                image_feat = image_feat.view(image_feat.size(0), -1)
            elif image_outputs.last_hidden_state.dim() == 3:
                image_feat = image_outputs.last_hidden_state.mean(dim=1)
            else:
                image_feat = image_outputs.last_hidden_state

        # Double check all features are flattened to 2D (fusion is impossible otherwise)
        if text_feat.dim() > 2:
            text_feat = text_feat.view(text_feat.size(0), -1)
        if image_feat.dim() > 2:
            image_feat = image_feat.view(image_feat.size(0), -1)

        # Concatenate features from both text and image 
        fused_features = torch.cat((text_feat, image_feat), dim=1)
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        return logits


# Training and evaluation 

def train_and_evaluate():
    # Replace these with your actual HuggingFace model IDs.
    text_model_id = "prodm93/twitter-text-sentiment-analyser"  
    image_model_id = "prodm93/resnet-50-finetuned-eurosat-albumentations"   

    # Initialise text model and tokeniser
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    text_model = AutoModel.from_pretrained(text_model_id)

    # Set up feature extraction
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_id)
    image_model = AutoModel.from_pretrained(image_model_id)

    num_classes = 3 # number of sentiment classes
    fusion_model = FusionClassifier(text_model, image_model, num_classes)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion_model.to(device)

    text_df = pd.read_json('processed_text.json')

    texts = text_df.text.values.tolist() 
    labels = text_df.labels.values.tolist()
    image_dataset = load_dataset("imagefolder", data_files={"train": "./Images.zip"})

    dataset = MultiModalDataset(texts, image_dataset, labels, tokenizer, feature_extractor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Set up optimiser and loss function for training
    optimizer = optim.Adam(fusion_model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Set up training loop
    fusion_model.train()
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            text_inputs, image_inputs, labels = batch
            labels = labels.to(device)

            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            optimizer.zero_grad()
            logits = fusion_model(text_inputs, image_inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.5f}")

    # Evaluation loop
    fusion_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            text_inputs, image_inputs, labels = batch
            labels = labels.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            logits = fusion_model(text_inputs, image_inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0, average='macro')
    rec = recall_score(all_labels, all_preds, zero_division=0, average='macro')
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='macro')

    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {acc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"F1 Score:  {f1:.5f}")

if __name__ == '__main__':
    train_and_evaluate()
