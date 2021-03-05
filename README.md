# Problem Statement Overview
This repository contains the code my team SPPU_AASM's submission for the WANLP Arsarcasm shared task 2021. The shared task consists of two research statements described below.

**Subtask 1 (Sarcasm Detection):** Identifying whether a tweet is sarcastic or not, this is a binary classification task.

**Subtask 2 (Sentiment Analysis):** Identifying the sentiment of a tweet and assigning one of three labels (Positive, Negative, Neutral), multiclass classification task.

# Dataset


**Label-wise distribution for Sarcasm Detection**
|Set|True|False|Total|
|:-------|:--------|:-------|:--------|
|Training|1734|8305|10039|
|Validation|434|2076|2510|
|Testing|821|2179|3000|
|Testing|2989|12559|15548|

<br>

**Label-wise distribution for Sentiment Identification**
|Set|Positive|Negative|Neutral|Total|
|:-------|:--------|:-------|:--------|:--------|
|Training|1744|3697|4598|10039|
|Validation|436|925|1149|2510|
|Testing|575|1677|748|3000|
|Testing|2755|6298|6495|15548|
<br>



# Proposed System
The proposed system is deep multi-chanel hybrid model which combines the word representations from pretrained AraBERT (A transformer-based language model trained on Arabic Wikipedia and news corpora), and pretrained Mazajak word embeddings (word2vec-based static word vectors trained on Twitter).

# Model Architecture
<img src="Model Architecture Diagram.png" height="480">


# Simulation Results
**Performance comparison of models for subtask-1. All metricscorrespond to the results on the sarcastic class.**

|Model|Accuracy|Precision|Recall|F1-Macro|F1-Sarcastic|
|:-------|:--------|:-------|:--------|:--------|:--------|
|Abu farah|1744|3697|4598|10039|
|AraBERT|0.85|0.75|0.78|0.77|0.52|
|AraBERT + CNN-BiLSTM|**0.86**|**0.76**|**0.78**|**0.77**|**0.62**|
|:-------|:--------|:-------|:--------|:--------|:--------|
|AraBERT + CNN-BiLSTM (Official results on test set)|0.7410|0.7031|0.7447|0.7096|0.6140|
<br>

