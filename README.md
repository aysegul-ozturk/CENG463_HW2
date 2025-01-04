# Binary Classification of Political Ideology and Orientation in Parliamentary Debates

This project is part of **CENG 463 - Introduction to NLP** and focuses on classifying political ideology and political orientation from parliamentary speech data. The implementation includes fine-tuning the XLM-RoBERTa-base model and experimenting with zero-shot learning using GPT-3 (text-davinci-003).

## Overview

The project aims to address two classification tasks:
1. **Political Ideology:** Identify whether the speaker's party leans left (0) or right (1).
2. **Political Orientation:** Determine if the speaker's party is in power (0) or in opposition (1).

## Methodology

### Dataset
- Data is derived from parliamentary speeches, with the Spanish dataset used for experimentation.
- **Class Imbalance:** Task 1's labels are balanced, while Task 2 has a slight imbalance (1:2 for label 1).

### Model Selection
- **Fine-Tuning:** XLM-RoBERTa-base, a multilingual transformer model.
- **Zero-Shot Learning:** GPT-3 (text-davinci-003) for causal inference.

### Training Details
- **Train-Test Split:** 90% training and 10% testing.
- **Hyperparameters:**
  - Learning Rate: `3e-5`
  - Batch Size: `4` (with gradient accumulation of 4 for an effective batch size of 16)
  - Epochs: `5`
  - Weight Decay: `0.01`
  - Warmup Steps: `500`
- **Zero-Shot Parameters:**
  - Temperature: `0.0` for deterministic outputs.

## Results

| Task | Model | Macro-Average F1-Score (Spanish) | Macro-Average F1-Score (English) |
|------|-------|----------------------------------|----------------------------------|
| Task 1: Ideology | Fine-Tuned | 0.56 | - |
|                   | Zero-Shot  | 0.37 | 0.41 |
| Task 2: Orientation | Fine-Tuned | 0.55 | - |
|                   | Zero-Shot  | 0.35 | 0.42 |

The fine-tuned XLM-RoBERTa-base model consistently outperformed the zero-shot GPT-3 model due to its ability to learn task-specific representations.

## Limitations
- Limited data for certain countries restricted model generalization.
- Computational constraints limited hyperparameter tuning.
- Zero-shot models require careful prompt engineering for complex tasks.

## Repository

You can find the complete implementation and code [here](https://github.com/aysegul-ozturk/CENG463_HW2).

---

This project is licensed under the Creative Commons License Attribution 4.0 International (CC BY 4.0).
