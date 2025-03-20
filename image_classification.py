import os
import argparse
import numpy as np
import torch
import albumentations as A
from datasets import load_dataset
import evaluate
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from huggingface_hub import login, HfFolder
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# class for preprocessing and transforming image data
class ImageTransformer:
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.setup_transforms()
        
    def setup_transforms(self):
        if "height" in self.image_processor.size:
            self.size = (self.image_processor.size["height"], self.image_processor.size["width"])
            self.crop_size = self.size
            self.max_size = None
        elif "shortest_edge" in self.image_processor.size:
            self.size = self.image_processor.size["shortest_edge"]
            self.crop_size = (self.size, self.size)
            self.max_size = self.image_processor.size.get("longest_edge")
        
        self.train_transforms = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.Normalize(),
        ])
    
    def preprocess_train(self, examples):
        examples["pixel_values"] = [
            self.train_transforms(image=np.array(image))["image"] for image in examples["image"]
        ]
        return examples
    
    def preprocess_val(self, examples):
        examples["pixel_values"] = [
            self.val_transforms(image=np.array(image))["image"] for image in examples["image"]
        ]
        return examples

# class for setting up image datasets
class ImageDatasetManager:
    
    def __init__(self, data_path, transformer):
        self.data_path = data_path
        self.transformer = transformer
        self.train_dataset = None
        self.val_dataset = None
        self.label2id = {}
        self.id2label = {}
        
    def load_dataset(self):
        dataset = load_dataset("imagefolder", data_files=self.data_path)
        return dataset
    
    def prepare_datasets(self):
        dataset = self.load_dataset()
        # Split dataset into training and validation sets
        splits = dataset["train"].train_test_split(test_size=0.2)
        self.train_dataset = splits['train']
        self.val_dataset = splits['test']
        
        # Set transformations
        self.train_dataset.set_transform(self.transformer.preprocess_train)
        self.val_dataset.set_transform(self.transformer.preprocess_val)
        
        # Extract labels
        labels = dataset["train"].features["label"].names
        for i, label in enumerate(labels):
            self.label2id[label] = i
            self.id2label[i] = label
        
        return self.train_dataset, self.val_dataset
    
    def get_num_labels(self):
        return len(self.id2label)

# class for image sentiment classification model
class ImageClassifier:
    
    def __init__(self, model_checkpoint, dataset_manager, use_cuda=True):
        self.model_checkpoint = model_checkpoint
        self.dataset_manager = dataset_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model = None
        self.trainer = None
        self.metric = evaluate.load("accuracy")
        
    def load_model(self):
        num_labels = self.dataset_manager.get_num_labels()
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_checkpoint,
            label2id=self.dataset_manager.label2id,
            id2label=self.dataset_manager.id2label,
            ignore_mismatched_sizes=True  # For fine-tuning an already fine-tuned checkpoint
        )
        self.model.to(self.device)
        return self.model
    
    def compute_metrics(self, eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=eval_pred.label_ids)
    
    def collate_fn(self, examples):
        images = []
        labels = []
        for example in examples:
            image = np.moveaxis(example["pixel_values"], source=2, destination=0)
            images.append(torch.from_numpy(image))
            labels.append(example["label"])
        
        pixel_values = torch.stack(images).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        return {"pixel_values": pixel_values, "labels": labels}
    
    def setup_trainer(self, training_args, image_processor):
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_manager.train_dataset,
            eval_dataset=self.dataset_manager.val_dataset,
            tokenizer=image_processor,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
        )
        return self.trainer
    
    def train(self):
        return self.trainer.train()
    
    def push_to_hub(self, repo_name=None, commit_message="Final model"):
        if repo_name is None:
            repo_name = f"{self.model_checkpoint.split('/')[-1]}-finetuned"
        
        return self.trainer.push_to_hub(commit_message=commit_message)

# class to handle functionalities involving HuggingFace Hub
class HuggingFaceHubManager:
    
    @staticmethod
    def login(token=None):
        if token:
            login(token=token, add_to_git_credential=True)
            return True
        
        # Check if already logged in
        if HfFolder.get_token():
            print("Already logged in to HuggingFace Hub.")
            return True
        
        # Interactive login
        try:
            login(add_to_git_credential=True)
            return True
        except Exception as e:
            print(f"Error logging in to HuggingFace Hub: {e}")
            return False
    
    @staticmethod
    def is_logged_in():
        return HfFolder.get_token() is not None


def parse_arguments():
    # set up CLI argument parsing
    parser = argparse.ArgumentParser(description="Train an image sentiment classification model")
    
    # Model and data arguments
    parser.add_argument("--model_checkpoint", type=str, default="microsoft/resnet-50",
                        help="Model checkpoint to use")
    parser.add_argument("--data_path", type=str, default="./Images.zip",
                        help="Path to the dataset (folder, zip, or tar file)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                        help="Learning rate")
    
    # CUDA arguments
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="Use CUDA if available")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help=""Set precision for training")
    
    # Hugging Face Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="HuggingFace Hub token")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Repo name on HuggingFace Hub")
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Handle Hugging Face Hub login
    if args.push_to_hub:
        hub_manager = HuggingFaceHubManager()
        if not hub_manager.login(args.hub_token):
            print("Failed to log in to HuggingFace Hub. Proceeding without pushing to Hub.")
            args.push_to_hub = False
    
    # Set up based on CUDA availability
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.multiprocessing.set_start_method('spawn') # common PyTorch error handling for multiprocessing 
        dataloader_pin_memory=False
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dataloader_pin_memory=True
        print("Using CPU")
    
    # Initialize image processor
    image_processor = AutoImageProcessor.from_pretrained(args.model_checkpoint)
    
    # Set up transformations
    transformer = ImageTransformer(image_processor)
    
    # Set up dataset manager and prepare datasets
    dataset_manager = ImageDatasetManager(args.data_path, transformer)
    train_dataset, val_dataset = dataset_manager.prepare_datasets()
    
    # Initialize classifier
    classifier = ImageClassifier(args.model_checkpoint, dataset_manager, use_cuda=use_cuda)
    model = classifier.load_model()
    
    # Set up training arguments
    model_name = args.model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        f"{model_name}-ft-image-classification",
        remove_unused_columns=False,
        dataloader_pin_memory=dataloader_pin_memory,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        # CUDA settings
        fp16=args.fp16 and use_cuda,  # Use mixed precision training when CUDA is available
        #dataloader_num_workers=4 if use_cuda else 0,  # Parallel data loading when CUDA is available
        # commented out, because parallel processing with dataloaders set to > 0 was causing too many errors
    )
    
    # Set up trainer and train
    trainer = classifier.setup_trainer(training_args, image_processor)
    trainer.train()
    
    # Save the model locally
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")
    
    # Push to hub if requested in arguments
    if args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        classifier.push_to_hub(args.hub_model_id)
        print(f"Model pushed to Hugging Face Hub at {args.hub_model_id or f'{model_name}-finetuned'}")


if __name__ == "__main__":
    main()