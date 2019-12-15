# Crop-stress-classification

This repository contains codes for the paper entitled <a href="https://arxiv.org/abs/1906.00454" target="_blank">"Classification of Crop Tolerance to Heat and Drought: A Deep Convolutional Neural Networks Approach"</a>. The paper was authored by Saeed Khaki, Zahra Khalilzadeh, and Lizhi Wang. In this paper, we proposed an unsupervised approach for classifying crops as either tolerant or susceptible to heat stress, drought stress, and combined drought and heat stress.

### Please cite our paper if you use our code. Thanks!
```
@article{khaki2019classification,
  title={Classification of Crop Tolerance to Heat and Drought: A Deep Convolutional Neural Networks Approach},
  author={Khaki, Saeed and Khalilzadeh, Zahra},
  journal={arXiv preprint arXiv:1906.00454},
  year={2019}
}

```


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
the data through annual Syngenta Crop Challenge. During the challenge, September 2018 to January 2019,
the data was open to the public. Researchers who wish to access the data may do so by contacting Syngenta
directly. We are not allowed to share the data due to non-disclosure agreement, sorry.




## Notice

### We have recenetly published a new paper titled <a href="https://arxiv.org/abs/1911.09045" target="_blank">"A CNN-RNN Framework for Crop Yield Prediction"</a> published in <a href="https://www.frontiersin.org/articles/10.3389/fpls.2019.01750/abstract" target="_blank"> Frontiers in Plant Science Journal</a>. This paper predicts corn and soybean yields based on weather, soil and management practices data. Reserachers can use the data from this paper using following <a href="https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction" target="_blank"> link</a>. We spend a lot of time gathering and cleaning the data from different publicly available sources. Please cite our papers if you use our data or codes. Thanks.

