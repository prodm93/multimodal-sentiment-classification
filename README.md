# Multimodal Sentiment Analysis

This project combines text and image modalities to create a model capable of accepting both as input and analysing its sentiment as positive, negative or neutral. The data used for proof-of-concept includes text and image data from Twitter.

## Overview

The repository contains tools for the following pipelines:
1. Text sentiment analysis using Transformers
2. Image sentiment analysis using ResNet vision model
3. Multimodal sentiment analysis combining text and image features

## Repository Structure
- `text_classification.py`: Script to preprocess text dataset and fine-tune text sentiment classification model
- `image_classification.py`: Script to preprocess images and training image sentiment classification model
- `multimodal_classification.py`: Script to combines text and image models using a concatenation layer for multimodal sentiment analysis
- `preprocessed_text.json`: File containing preprocessed Twitter text data ready for the fine-tuning process
- `requirements.txt`: Packages required to run the scripts

## Installation

```bash
# Clone the repository (make sure you have git installed)
git clone https://github.com/yourusername/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis

# Create a virtual environment for the project (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Text Sentiment Analysis

```bash
# Process text data and fine-tune a text sentiment model
python text_classification.py --model_checkpoint "lxyuan/distilbert-base-multilingual-cased-sentiments-student" \
                           --batch_size 8 \
                           --num_epochs 5 \
                           --learning_rate 2e-5 \
                           --save_processed True

# For preprocessing only, without training 
python text_classification.py --preprocess_only --save_processed True
```

### Image Sentiment Analysis

For Twitter dataset analysis, <complete this later
```bash
# Train an image sentiment model
python image_classification.py --model_checkpoint "microsoft/resnet-50" \
                            --data_path "./Images.zip" \
                            --batch_size 16 \
                            --num_epochs 5 \
                            --learning_rate 0.05
```

### Multimodal Sentiment Analysis

```bash
# Run multimodal concatenation and train fused nodel after training both text and image models
python multimodal_classification.py
```
