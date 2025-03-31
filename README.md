# Fake News Detection using Neural Networks and Hybrid Intelligent Systems

This project involves implementing a neural network to classify news articles as real or fake, along with proposing a hybrid intelligent system (ANFIS) to improve classification accuracy. The system is trained on a balanced dataset of 10,000 news articles (5,000 real and 5,000 fake) from Kaggle.

## Overview

The project focuses on building and evaluating a Bidirectional LSTM neural network for fake news detection, achieving 99% accuracy. It also proposes an Adaptive Neuro-Fuzzy Inference System (ANFIS) to handle ambiguous cases. Key components include:

- Text preprocessing (tokenization, padding)
- Neural network architecture (Embedding + BiLSTM + Dropout)
- Performance evaluation (accuracy, precision, recall)
- Hybrid system design (LSTM + Fuzzy Logic integration)

## How To Run

1. Clone the Repository 

```
git clone https://github.com/Karma151205/news-classifier-ANN-project.git
cd news-classifier-ANN-project
```

2. Install Required Dependencies 

```
pip install -r requirements.txt
```

3. Run the Script 

```
python A2.py
```
