# Stock Market Analysis and Forecasting Using Deep Learning

## Overview
This project involves the analysis and forecasting of stock market prices using deep learning techniques, specifically utilizing GRU (Gated Recurrent Unit) in PyTorch.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Analysis](#analysis)
- [Forecasting](#forecasting)
- [GRU Model](#gru-model)
- [Results](#results)
- [Blog](#blog)
- [License](#license)

## Introduction
A stock market, equity market, or share market is the aggregation of buyers and sellers of stocks (also called shares), which represent ownership claims on businesses. These may include securities listed on a public stock exchange, as well as stock that is only traded privately, such as shares of private companies sold to investors through equity crowdfunding platforms. Investment in the stock market is often done via stockbrokerages and electronic trading platforms, usually with an investment strategy in mind.

Stock prediction has always been a challenging problem for statistics experts and finance professionals. The main goal is to buy stocks that are likely to increase in price and sell stocks that are probably going to fall. There are two main methods for stock market prediction:
1. **Fundamental Analysis:** Relies on a company's technique and fundamental information like market position, expenses, and annual growth rates.
2. **Technical Analysis:** Concentrates on previous stock prices and values.

In the first part of our project, we will analyze the data, and in the second part, we will forecast the stock market.

## Dataset
We will be using historical stock data from the following companies:
- Google
- Microsoft
- IBM
- Amazon

## Analysis
### Microsoft
- The "High" value shows a very slowly increasing straight line.
- In 2009, the "High" value was under the mean for a long time, indicating some loss.

### IBM vs Amazon
- IBM's "High" value and Amazon's "High" value started from approximately the same stage. Amazon's "High" value was a bit lower initially, but after 2012, Amazon's "High" value started to increase exponentially, while IBM's "High" value saw a slight drop.
- Since 2016, there has been significant competition between Google's "High" value and Amazon's "High" value. In 2018, Amazon's "High" value surpassed Google's "High" value.

### Google
- There is a very slow increasing trend until 2012, followed by an exponential high trend. High seasonality is observed.

## Forecasting
Time series forecasting uses historical values and associated patterns to predict future activity. This includes:
- Trend analysis
- Cyclical fluctuation analysis
- Seasonality issues

As with all forecasting methods, success is not guaranteed.

## GRU Model
Gated Recurrent Unit (GRU) is essentially a simplified version of Long Short-Term Memory (LSTM). It serves the same role in the network but is somewhat simpler due to fewer gates and weights. It has two gates:
- **Update Gate:** Controls the information flow from the previous activation and the addition of new information.
- **Reset Gate:** Inserted into the candidate activation.

Since GRU does not have an output gate, there is no control over the memory content.

## Results
Detailed results and visualizations will be added here.

