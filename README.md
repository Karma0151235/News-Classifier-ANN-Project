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


</br>
Note: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place it in the `data/` folder before execution.

## Dataset 
The dataset consists of two CSV files:
- `True.csv`: 21,417 real news articles
- `Fake.csv`: 23,502 fake news articles

Each article contains:
- `Title`: News headline
- `Text`: Article content
- `Subject`: Category (e.g., politics)
- `Date`: Publication date

Preprocessing steps include:
1. Combining title and text
2. Balancing classes (5,000 samples each)
3. Tokenization and sequence padding (200 words)

## Technologies Used

**Python**: Primary implementation language <br/>
**TensorFlow/Keras**: Neural network construction <br/>
**Pandas/Numpy**: Data manipulation <br/>
**Scikit-learn**: Evaluation metrics <br/>
**Matplotlib/Seaborn**: Visualizations <br/>

## Results
The model achieves:

| Metric        | Fake News | True News | Weighted Avg |
|---------------|-----------|-----------|--------------|
| Precision     | 98%       | 100%      | 99%          |
| Recall        | 100%      | 98%       | 99%          |
| F1-score      | 99%       | 99%       | 99%          |
| Accuracy      |           |           | 99%          |

## Visualizations
**Model Accuracy/Loss**: Training vs validation curves <br/>
**Confusion Matrix**: TP/TN/FP/FN distribution <br/>
**ANFIS Architecture**: Hybrid system design schematic <br/>

## Current Issues
1. Handling ambiguous news articles
2. Potential overfitting despite dropout layers
3. Manual hyperparameter tuning limitations

Next steps include implementing the proposed ANFIS hybrid system and automated hyperparameter optimization.
