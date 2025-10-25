# Text Summarization Using BART and T5 Models

A comprehensive comparative study of BART and T5 transformer models for abstractive text summarization on the CNN/Daily Mail dataset, exploring pre-trained model performance and advanced fine-tuning techniques.

## Overview

This project investigates three key aspects of text summarization using Large Language Models (LLMs):

1. How do pre-trained models initially perform on the CNN/Daily Mail dataset?
2. What techniques can be used to effectively re-train LLMs on a relatively small dataset?
3. How do different models react to the training process, and what are their performances after fine-tuning?

## Models Evaluated

Six different variations of BART and T5 models were tested and fine-tuned:

| Model | Layers | Attention Heads | Hidden Size | Parameters | Vocabulary Size |
|-------|--------|----------------|-------------|------------|----------------|
| **BART-BASE** | 12 | 12 | 768 | 139M | 50,265 |
| **BART-LARGE** | 12 | 16 | 1024 | 406M | 50,265 |
| **BART-LARGE-CNN** | 12 | 16 | 1024 | 406M | 50,265 |
| **T5-SMALL** | 6 | 8 | 512 | 60M | 32,128 |
| **T5-BASE** | 12 | 12 | 768 | 220M | 32,128 |
| **T5-LARGE** | 24 | 16 | 1024 | 770M | 32,128 |

### Model Descriptions

- **BART (Bidirectional and Auto-Regressive Transformers)**: A Seq-to-Seq model with a bidirectional encoder (BERT-inspired) and an autoregressive decoder (GPT-inspired)
- **T5 (Text-To-Text Transfer Transformer)**: A versatile transformer-based model that frames all NLP tasks as text-to-text problems

## Dataset

**CNN/Daily Mail Dataset (v3.0.0)**
- One of the most popular datasets for text summarization tasks
- Contains 311,971 long articles from CNN and Daily Mail
- Each article includes a human-written summary

## Project Structure

```
├── 1_code_data_visualization.ipynb    # Data exploration and visualization
├── 2_code_text_summarization.ipynb    # Model training and evaluation
├── 3_code_experiment_analysis.ipynb   # Results analysis and comparison
└── README.md
```

## Notebooks Description

### 1. Data Visualization (`1_code_data_visualization.ipynb`)
- Load and preprocess CNN/Daily Mail dataset
- Tokenization with BART and T5 tokenizers
- Comprehensive data visualization:
  - Word clouds of most common tokens
  - Top 20 common tokens analysis
  - Distribution of article lengths
  - Distribution of summary lengths
  - Article-summary length correlation analysis
- Qualitative analysis of sample articles and summaries

### 2. Text Summarization (`2_code_text_summarization.ipynb`)
- **Experiment Design**: 6-stage pipeline from data preparation to inference
- **Pre-training Evaluation**: Baseline performance metrics
- **Model Fine-tuning**:
  - Hyperparameter tuning
  - Advanced fine-tuning techniques (layer freezing, discriminative fine-tuning)
  - Regularization methods
- **Performance Evaluation**: ROUGE score computation
- **Pipeline Building**: Create production-ready summarization pipeline
- **Real-world Testing**: Evaluate on latest news articles

### 3. Experiment Analysis (`3_code_experiment_analysis.ipynb`)
- Pre-training evaluation results comparison
- Training and validation metrics visualization
- Model training time analysis
- Performance evaluation across all models
- Improvement rates calculation
- Ablation studies on fine-tuning techniques

## Key Findings

### Pre-training Performance

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Mean Length |
|-------|---------|---------|---------|------------|-------------|
| BART-BASE | 30.41 | 11.96 | 18.92 | 25.04 | 117.0 |
| BART-LARGE | 28.68 | 9.98 | 18.64 | 24.40 | 117.1 |
| BART-LARGE-CNN | 40.37 | 16.99 | 28.98 | 34.36 | 50.2 |
| T5-SMALL | 28.83 | 8.67 | 20.12 | 24.28 | 45.1 |
| T5-BASE | 35.60 | 14.49 | 26.49 | 30.99 | 42.7 |
| T5-LARGE | 31.59 | 11.04 | 22.23 | 26.75 | 31.5 |

### Key Insights

1. **T5 vs BART**: T5 models generally outperform BART models due to their versatile text-to-text architecture
2. **Pre-training Matters**: BART-LARGE-CNN shows the best performance as it was pre-trained on the CNN/Daily Mail dataset
3. **Model Size**: Larger models generally achieve better results due to increased capacity and expressive ability
4. **Generation Length**: Output length significantly impacts performance, particularly evident in T5-LARGE

## Advanced Techniques Explored

- **Layer Freezing**: Selectively freeze lower layers to reduce overfitting
- **Discriminative Fine-tuning**: Apply different learning rates to different layers
- **Hyperparameter Tuning**: Optimize batch size, learning rate, and training epochs

## Requirements

```
torch
datasets
transformers
rouge-score
accelerate
numpy
pandas
matplotlib
seaborn
wordcloud
nltk
```

## Installation

```bash
pip install torch datasets transformers rouge-score accelerate
pip install numpy pandas matplotlib seaborn wordcloud nltk
```

## Usage

### 1. Data Visualization
```python
jupyter notebook 1_code_data_visualization.ipynb
```

### 2. Model Training
```python
jupyter notebook 2_code_text_summarization.ipynb
```

### 3. Results Analysis
```python
jupyter notebook 3_code_experiment_analysis.ipynb
```

## Evaluation Metrics

The project uses ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metrics:

- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap between generated and reference summaries
- **ROUGE-L**: Longest common subsequence between generated and reference summaries
- **ROUGE-Lsum**: ROUGE-L computed on summary-level
