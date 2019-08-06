# Crop-stress-classification

This repository contains codes for the paper entitled <a href="https://arxiv.org/abs/1906.00454" target="_blank">"Classification of Crop Tolerance to Heat and Drought: A Deep Convolutional Neural Networks Approach"</a>. The paper was authored by Saeed Khaki, Zahra Khalilzadeh, and Lizhi Wang. In this paper, we proposed an unsupervised approach for classifying crops as either tolerant or susceptible to heat stress, drought stress, and combined drought and heat stress.


## Getting Started 

Please install the following packages in Python3:

- numpy
- tensorflow
- matplotlib
- scikit-learn
- time


## Dimension of Input Data

- The weather and soil data were used in this paper. You should load your data and then run the model.
- The weather and soil data both have `m-by-n` array format where m is the number of environments (observations) and n is the number of features.
- the yield stress (response) is `m-by-1` array.


## The order of codes:

- `CNN-Drought.py`, `CNN-heat.py`, `CNN-combined_Drought_Heat.py`
- `stress_metric_extraction.py`
-  `PCA_part.py`
- `Hybrid Classification.py`




##  Data Availability Statement 

The data analyzed in this study was provided by Syngenta for 2019 Syngenta Crop Challenge. We accessed
the data through annual Syngenta Crop Challenge. During the challenge, September 2017 to January 2018,
the data was open to the public. Researchers who wish to access the data may do so by contacting Syngenta
directly.
