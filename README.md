# SentimentLoRA

# LoRA Text Classification

A parameter-efficient fine-tuning implementation for sentiment analysis using LoRA (Low-Rank Adaptation) on the IMDB dataset.

## Description

This project demonstrates how to apply Parameter-Efficient Fine-Tuning (PEFT) using LoRA to adapt a pre-trained language model (DistilBERT) for sentiment classification tasks. It uses a small subset of the IMDB movie reviews dataset to demonstrate the efficiency of LoRA for fine-tuning large language models with minimal computational resources.

The implementation:
- Uses a truncated IMDB dataset (1,000 examples for training and validation)
- Applies LoRA to efficiently fine-tune only a small subset of model parameters
- Trains a sentiment classifier to categorize movie reviews as positive or negative
- Provides utilities for model evaluation and inference
- Includes functionality to save and load models from the Hugging Face Hub

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- Datasets
- Evaluate
- NumPy
- Hugging Face Hub

## Installation

```bash
pip install transformers peft datasets evaluate torch numpy huggingface_hub
