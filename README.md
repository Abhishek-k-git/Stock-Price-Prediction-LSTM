## Stock Price Prediction Using LSTM

### Table of content
* [Overview](#Overview)
* [Folder Structure](#Folder-Structure)
* [Setup](#Setup)
* [Deployment](#Deployment)
* [Go to App](https://stock-price-prediction-lstm.streamlit.app/)

### Overview

This project encompasses the prediction of stock closing prices utilizing **Python** and the **yfinance** library. The model is trained by leveraging the capabilities of the **Long Short-Term Memory (LSTM)** layer in **Keras**. The predictive model is then seamlessly hosted through Streamlit, rendering it user-oriented and easily accessible. This project serves to provide a comprehensive and interactive platform for accurate stock market prediction.

- Keras is a high-level, deep learning API developed by Google for implementing neural networks. It is written in Python and is used to make the implementation of neural networks easy. It also supports multiple backend neural network computation.
- The *Long Short-Term Memory, or LSTM*, network is a type of Recurrent Neural Network (RNN) designed for sequence problems.
- *yfinance* is a Python library that allows us to easily download financial data from Yahoo Finance.

### Folder-Structure

```
├── Application (live on Streamlit)
    ├── LSTM Model.ipynb - Contains Data Processing training/testing and model building
    ├── app.py - contains code for streamlit app 
    └── keras_model.h5 - contains model build by keras
```

#### Setup

**Installations**

```
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install pandas_datareader
!pip install yfinance
!pip install streamlit
!pip install tensorflow
!pip install keras
```
**import**

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
```

**read data**
```
yf.pdr_override()
df = pdr.get_data_yahoo(ticker, start_date, end_date)
```

**run streamlit app locally**
```
streamlit run app.py
```
    
### Deployment

*Step 1: Version Control with Git*
Using Git for version control, add your application and the requirements.txt file into a new repository. This will make deploying your app easier in later steps and allow others to contribute to your app.

*Step 2: Deployment with GitHub and Streamlit Sharing*
To deploy your Streamlit app, you can use Streamlit Sharing, which requires a GitHub account. Go to Streamlit Sharing dashboard, link it to your Github account, then enter the repo, branch, and path to your app, then click 'Deploy'. Streamlit will deploy your app and provide a link.

*Step 3: Configure the Streamlit Sharing dashboard *
Customize your app by updating the name, adding a thumbnail, or changing the privacy settings. Streamlit Sharing also provides features for secret management, app performance monitoring, and app logs.

```diff
! Note: In Python, a requirements.txt file is a type of file that usually stores information about all the libraries, modules, and packages in itself that are used while developing a particular project. It should be included, while deploying on streamlit live server.
```
