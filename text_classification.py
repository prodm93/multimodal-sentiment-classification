import os
import argparse
import numpy as np
import torch
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasets import Dataset, load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from huggingface_hub import login, HfFolder
from typing import Dict, List, Optional

# class for preprocessing text data
class TextPreprocessor:
 
    def __init__(self):
        # Download required NLTK resources
        for resource in ['stopwords', 'punkt', 'punkt_tab', 'wordnet']:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = text.lower()  # lowercase all text
        text = re.sub(f'[{string.punctuation}]', '', text)  # remove punctuation
        
        # use tokeniser to break text down into chunks
        words = word_tokenize(text)
        
        # remove stopwords and lemmatise all words
        lemmas = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(lemmas)

# class for setting up text datasets
class TextDatasetManager:
    
    def __init__(self, data_path=None, preprocessor=None):
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.tokenized_train = None
        self.tokenized_val = None
        self.label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        self.processed_df = None  

    # load dataset directly from Kaggle
    def load_kaggle_dataset(self, kaggle_dataset="dunyajasim/twitter-dataset-for-sentiment-analysis",
                            file_path="LabeledText.xlsx"):
        
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                kaggle_dataset,
                file_path
            )
            
            # Map labels and rename columns for downstream model handling
            if "LABEL" in df.columns:
                df["LABEL"] = df["LABEL"].map(self.label_mapping)
                df = df.rename(columns={"LABEL": "labels"})
            
            if "Caption" in df.columns:
                df = df.rename(columns={"Caption": "text"})
                
            # Drop unnecessary columns
            if "File Name" in df.columns:
                df.drop("File Name", axis=1, inplace=True)
                
            return df
        except Exception as e:
            print(f"Error loading Kaggle dataset: {e}")
            return None
    
    # option to load dataset from local file
    def load_local_dataset(self):
        try:
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                df = pd.read_excel(self.data_path)
            elif self.data_path.endswith('.json'):
                df = pd.read_json(self.data_path)
            else:
                raise ValueError("Unsupported file format. Please provide a valid CSV, Excel, or JSON file.")
                
            return df
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            return None
    
    def preprocess_dataset(self, df):
        # preprocess data
        if self.preprocessor:
            print("Preprocessing text data...")
            df['text'] = df['text'].apply(self.preprocessor.preprocess_text)
        self.processed_df = df  # Store processed dataframe
        return df
    
    def save_processed_data(self, output_path="./processed_text.json"):
        # save processed data to JSON file
        if self.processed_df is not None:
            try:
                print(f"Saving processed data to {output_path}...")
                self.processed_df.to_json(output_path, orient='records')
                print(f"Data successfully saved to {output_path}")
                return True
            except Exception as e:
                print(f"Error saving processed data: {e}")
                return False
        else:
            print("No processed data available to save.")
            return False
    
    def prepare_datasets(self, test_size=0.2, seed=42):
        # Split data into training and validation sets
        if self.data_path is not None:
            df = self.load_local_dataset()
        else:
            df = self.load_kaggle_dataset()
            
        if df is None:
            print("Please provide a valid data path or Kaggle dataset.")
            return None, None
            
        # Preprocess dataset
        df = self.preprocess_dataset(df)
        
        # Convert to datasets format
        self.dataset = Dataset.from_pandas(df)
        
        # Split dataset
        splits = self.dataset.shuffle(seed=seed).train_test_split(test_size=test_size)
        self.train_dataset = splits['train']
        self.val_dataset = splits['test']
        
        return self.train_dataset, self.val_dataset
    
    def tokenize_datasets(self, tokenizer):
        # Tokenise datasets
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True)
        
        self.tokenized_train = self.train_dataset.map(tokenize_function, batched=True)
        self.tokenized_val = self.val_dataset.map(tokenize_function, batched=True)
        
        return self.tokenized_train, self.tokenized_val

# class for sentiment analysis for text
class SentimentClassifier:
    
    def __init__(self, model_checkpoint="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                 num_labels=3, use_cuda=True):
        self.model_checkpoint = model_checkpoint
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model_and_tokenizer(self):
        # Initialise model and tokeniser 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        return self.model, self.tokenizer
    
    def compute_metrics(self, eval_pred):
        # set up metrics for training evaluation
        metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        
        return {**accuracy, **f1}
    
    def setup_trainer(self, train_dataset, val_dataset, training_args):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        return self.trainer
    
    def train(self):
        return self.trainer.train()
    
    def evaluate(self):
        return self.trainer.evaluate()
    
    def push_to_hub(self, repo_name=None, commit_message="Final model"):
        # function to push model to huggingface hub
        if repo_name is None:
            repo_name = f"{self.model_checkpoint.split('/')[-1]}-finetuned-sentiment"
        
        return self.trainer.push_to_hub(commit_message=commit_message)

# class to handle functionalities involving HuggingFace Hub
class HuggingFaceHubManager:
    
    @staticmethod
    def login(token=None):
        if token:
            login(token=token, add_to_git_credential=True)
            return True
        
        # check if already logged in
        if HfFolder.get_token():
            print("Already logged in to Hugging Face Hub.")
            return True
        
        try:
            login(add_to_git_credential=True)
            return True
        except Exception as e:
            print(f"Error logging in to Hugging Face Hub: {e}")
            return False
    
    @staticmethod
    def is_logged_in():
        return HfFolder.get_token() is not None


def parse_arguments():
    # set up CLI argument parsing
    parser = argparse.ArgumentParser(description="Fine-tune a text sentiment analysis model")
    
    # Model and data arguments
    parser.add_argument("--model_checkpoint", type=str, default="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
                        help="Model checkpoint to use")
    parser.add_argument("--data_path", type=str, 
                        help="Path to the dataset file (CSV, Excel, or JSON), if using local file")
    parser.add_argument("--save_processed", action="store_true", default=True,
                        help="Option to save processed data as JSON")
    parser.add_argument("--processed_output_path", type=str, default="./processed_text.json",
                        help="Path to save processed data as JSON")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for regularisation")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of dataset to use for validation")
    
    # CUDA arguments
    parser.add_argument("--use_cuda", action="store_true", default=True,
                        help="Use CUDA if available")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Set precision for training")
    
    # Hugging Face Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="HuggingFace Hub token")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Repo name on HuggingFace Hub, ensure prefixed username is included")
    
    # Preprocessing only mode
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only preprocess and save data without training")
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Handle Hugging Face Hub login
    if args.push_to_hub:
        hub_manager = HuggingFaceHubManager()
        if not hub_manager.login(args.hub_token):
            print("Failed to login to HuggingFace Hub. Proceeding without pushing to hub.")
            args.push_to_hub = False
    
    # Set up based on CUDA availability
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        dataloader_pin_memory=False
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dataloader_pin_memory=True
        print("Using CPU")
    
    # Initialize text preprocessor
    preprocessor = TextPreprocessor()
    
    # Set up dataset manager and prepare datasets
    dataset_manager = TextDatasetManager(args.data_path, preprocessor)
    train_dataset, val_dataset = dataset_manager.prepare_datasets(test_size=args.test_size)
    
    if train_dataset is None or val_dataset is None:
        print("Failed to prepare datasets. Exiting.")
        return
    
    # Save processed data if requested in arguments
    if args.save_processed:
        output_path = args.processed_output_path if args.processed_output_path is not None else "./processed_text.json"
        # Add .json extension if not present
        if not output_path.endswith('.json'):
            output_path += '.json'
        dataset_manager.save_processed_data(output_path)
        
        if args.preprocess_only:
            print("Preprocessing completed. Exiting...")
            return
    
    # Initialize classifier
    classifier = SentimentClassifier(args.model_checkpoint, num_labels=3, use_cuda=use_cuda)
    model, tokenizer = classifier.load_model_and_tokenizer()
    
    # Tokenize datasets
    tokenized_train, tokenized_val = dataset_manager.tokenize_datasets(tokenizer)
    
    # Set up training arguments
    model_name = args.model_checkpoint.split("/")[-1]
    repo_name = args.hub_model_id or "prodm93/twt-sentiment-analysis"
    
    training_args = TrainingArguments(
        output_dir=repo_name,
        dataloader_pin_memory=dataloader_pin_memory,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=args.push_to_hub,
        hub_model_id=repo_name,
        # CUDA settings
        fp16=args.fp16 and use_cuda,  # Use mixed precision training when CUDA is available
        #dataloader_num_workers=4 if use_cuda else 0,  # Parallel data loading when CUDA is available
        # commented out, because parallel processing with dataloaders set to > 0 was causing too many errors
    )
    
    # Set up trainer and train
    trainer = classifier.setup_trainer(tokenized_train, tokenized_val, training_args)
    print("Starting training...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = classifier.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model locally
    trainer.save_model()
    print(f"Model saved to {training_args.output_dir}")
    
    # Push to Hub if requested
    if args.push_to_hub:
        print("Pushing model to HuggingFace Hub...")
        classifier.push_to_hub(args.hub_model_id)
        print(f"Model pushed to HuggingFace Hub at {repo_name}")


if __name__ == "__main__":
    main()