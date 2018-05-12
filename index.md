<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<script type="text/javascript">
 MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [ ['$','$'], ['\\(','\\)'] ]
  }
});
</script>

[fig1]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/price.png
[fig2]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/rsi-mfi_medium.png
[fig3]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/ppo_medium.png
[fig4]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/nn.png
[fig5]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/profits.png
[fig7]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/d_sin_medium.png
[fig8]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/q_sin_medium.png
[fig9]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/bh_medium.png
[fig10]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/demo_profitabilities_medium.png

[fig6]: d_7200.png
[fig6]: q_7200.png
[fig6]: r.png


# I. Definition

## Project Overview

Algorithmic trading attracted many researchers in recent years. Their goal is to create software that
is able to trade autonomously and profitably. Most of the works and projects are focused on
traditional stock markets ([D. W. Lu](https://arxiv.org/abs/1707.07338v1), [Wang et al](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/5/5f/Dtq.pdf)) or fiat currency markets while this project is focused on cryptocurrency
market.

A cryptocurrency is a digital asset designed to work as a medium of exchange using cryptography to
secure the transactions, to control the creation of additional units, and to verify the transfer of
assets. [Cryptocurrencies](https://en.wikipedia.org/wiki/Cryptocurrency) are classified as a subset of digital currencies and are also classified as a subset of alternative currencies and virtual currencies.

Cryptocurrency market refers to the collection of markets and exchanges where trading of
cryptocurrencies and [tokens](https://en.wikipedia.org/wiki/Initial_coin_offering) takes place, either through formal exchanges or over-the-counter markets.

Cryptocurrency market is much more volatile than stock market. Volatility of stock market is
summarized by [VIX](https://finance.yahoo.com/quote/%5EVIX/) index and is around 16% a year for now. Volatility of cryptocurrency market is not summarized by commonly accepted index. But still there are quite illustrative indices e.g. [SIFR Volatility Index](https://www.sifrdata.com/cryptocurrency-volatility-index/), it is around 150% for now.

Increased volatility creates more buy and sell opportunities and therefore such market is more
[suitable](https://www.investopedia.com/ask/answers/010915/volatility-good-thing-or-bad-thing-investors-point-view-and-why.asp) for short-term technical analysis based trading.

Stock and fiat currencies markets differs from cryptocurrency market in terms of regulation, volatility and risk of market manipulation but the basic idea is the same. Therefore approaches that were discussed in stock market related works should also be suitable for cryptocurrency market.

### Existing research
There is a work dedicated to autonomous trading system based on Deep Q-learning. This work is quite inspiring because authors shown that Deep Q-learning produced the most significant results and their system was above baselines.

### Relevance
There are plenty of reasons why is this research relevant:
 - cryptocurrency market is quite young and there are little or no works related to its
automation
 - the market is easy to enter and there is a lot of freely available data
 - exchanges are open 24/7 and good conditions to buy or sell can be missed by a human trader

### Datasets
To make trading decisions an automatic trading system should acquire the most recent data of currency prices and volumes. A major exchanges provide APIs that make this data freely available at any time. For example to retrieve [data](https://poloniex.com/public?command=returnChartData&currencyPair=BTC_ETH&start=1515542400&end=1517443200&period=86400) of prices and volumes of Bitcoin to Etherium trades on Poloniex for January 2018 only desired period (e.g. 1 day, 4 hours etc.) and timestamps for start and end of the month are required. Data is returned in JSON and contains open, close, high and low prices for the period and traded volumes. This data is sufficient for calculating all indicators utilized for decision making.

## Problem Statement
The problem to solve is to automate trading process by means of automated trading system.

Trading is the process of buying and selling cryptocurrency on specialized exchange over the Internet. There are a lot of [exchanges](https://coinmarketcap.com/exchanges/volume/24-hour) but for the sake of clarity and simplicity the project will use [Poloniex](https://poloniex.com) and its API.

The purpose of trading is to increase total price of assets hold by trader. The price can be nominated in fiat currency or in any cryptocurrency of choice. Trader is the person who commits buys and sells of digital assets or handles automated trading system that commits these actions.

Automated trading system should be able to recognize buy and sell signals and recognize situations when to hold. Signals to buy, sell or hold should be recognized by means of machine learning. To recognize these signals machine learning part of the system should be able to infer state of the market by processing input features. Features are produced by means of technical analysis and technical indicators in particular. Technical analysis is an analysis methodology for forecasting the direction of prices through the study of past market data, primarily [price and volume](https://en.wikipedia.org/wiki/Technical_analysis). In technical analysis, a technical indicator is a mathematical calculation based on historic price and volume that aims to forecast [financial market direction](https://en.wikipedia.org/wiki/Technical_indicator).

Therefore the input of the system is the state of market and the output is one particular action. The state is defined by a set of features related to price and volume of a given asset. Actions are 'buy', 'sell' or 'hold'. The problem is a classification one because it is essentially a mapping of a state to an action.

### Problem structure
To solve the problem under discussion it should be broken down into following parts:
 - feature selection
 - creation of isolated environment
 - baseline and primary learning algorithm implementation
 - learning process establishment
 - building an application

### Feature selection
To infer a state of an environment a set of features should be introduced. Each feature should bring new knowledge to the system. Therefore it makes sense to use technical indicators responsible for different properties of the environment (price change, volume change, oscillation) as features. Technical indicators are calculated using a sequence of values and therefore illustrate not only a current state of the environment but also its change in time.

#### Creation of isolated environment
Isolated environment is required until profitability of the system is proven. Balances of trader and exchange should be mocked and no real value should be involved in trades. However price and volume data should be taken from a real exchange. Implementing real stock trading is beyond the scope of this work.

#### Baseline and primary learning algorithm implementation
As it was discussed in Capstone Proposal Q-Learning and Deep Q-Learning algorithms will be implemented in this project. It is expected that Deep Q-Learning algorithm shows better performance that is why it is considered as primary while Q-Learning algorithm is considered as a baseline. Both algorithms use the same set of features and same learning process. Their core components are the only difference. Q-Learning algorithm uses table for classification and its deep counterpart uses neural network.

#### Learning process establishment
Learning process is the sequence of actions dedicated to train agents. Learning process for both types of agents of this project should be the same. Initially agents are trained using data stored in advance. Running agents are trained periodically after some specified delay in order to adapt to changing environment. Full retraining with dropping of trained models also should occur once in a while to prevent agents to accumulate obsolete knowledge.

#### Building an application
Algorithms and all code for their maintenance should be wrapped up in web-application in order to provide visual representation of their work and state.

## Metrics

### Profitability

Profitability is a price difference between initial holdings of trader and his/her holdings at a given moment after some trades were executed. Profitability should be denoted in percents.

Price of holdings for this particular project should be nominated in Bitcoin (BTC). It is done because a market the system is trying to beat is nominated in BTC. Also Bitcoin has many years' uptrend and is the most widely acceptable and the most recognized cryptocurrency. System's profitability should be calculated as follows:

$$ P = p_0 p_n \cdot 100 $$

where
 - $p_0$ is a price of holdings in Bitcoin at the start of training period
 - $p_n$ is a price of holdings in Bitcoin at the end of training period

### Accuracy and recall
To calculate accuracy and recall a convention should be introduced. Let 'sell' be negative and 'buy' be positive. Then if 'sell' is profitable then it is true negative. If 'sell' is not profitable then it is false negative. If 'buy' is profitable then it is true positive. If 'buy' is not profitable then it is false positive. Profitability of the action will be calculated on period between a given action and the next one. Having these values a confusion matrix will be built so accuracy and recall will be calculated. These statistical measurements provide additional performance assessment of the system and should be used to estimate classification performance gains or losses caused by changes in algorithms.


# II. Analysis

## Data Exploration

Historical data of prices and volumes of cryptocurrencies is a dataset for solving the problem. This data is available via public APIs of [cryptocurrency exchanges](https://poloniex.com/support/api/). Usually years of data are freely available.

Trading system will support multiple currencies. For the purposes of the project datasets for the
following currency pairs for one year will be used:

 - [BTC-ETH](https://goo.gl/Q16q98)
 - [BTC-XMR](https://goo.gl/WHkkP5)
 - [BTC-DASH](https://goo.gl/S8mRfR)
 - [BTC-LTC](https://goo.gl/hmvFum)
 - [BTC-XRP](https://goo.gl/fMRxcs)

These pairs were selected according to their Sharpe ratio, volume of trades on exchange and overall market capitalization.

There are different datasets for different time periods. Time period in terms of candlestick chart is a time covered by one candle. Poloniex exchange, and therefore all datasets, supports seven periods (in seconds): 300, 900, 1800, 7200, 14400, 86400. The last three will be used in the project for making classifications. The first one will be used to get the latest price for a given date.

Dataset for four hours period contains 2191 entries. Every entry is essentially a candle of candlestick chart and contains date as timestamp, high (the highest price in the candle), low (the lowest price in the candle), open (start price of the candle), close (end price of the candle), volume (of trades of base currency, in case of BTC-ETH it is BTC), quote volume (of trades of quoted currency, in case of BTC-ETH it is ETH) and weighted average of price.

Table 1. Example of data, BTC-ETH candlestick chart data, 4 hours period

|date|close|high|low|open|quoteVolume|volume|weightedAverage|
|-|-|-|-|-|-|-|-|
|1479528000| 0.012692| 0.012920| 0.012671| 0.012760| 45453.92| 582.84| 0.012822|
|1479542400| 0.012891| 0.013099| 0.012685| 0.012686| 63869.23| 824.77| 0.012913|
|1479556800| 0.012902| 0.012926| 0.012870| 0.012900| 11916.39| 153.71| 0.012899|
|1479571200| 0.012827| 0.012994| 0.012815| 0.012919| 28821.06| 372.02| 0.012908|
|1479585600| 0.012820| 0.012996| 0.012783| 0.012839| 22480.96| 290.20| 0.012908|

The data is of high quality and does not need to be cleaned up, however it is not suitable for the solution. Solution will use statistical measures of the data i.e. technical indicators so preprocessing is required.

## Exploratory Visualization
Solution of the problem requires time series price and volume data to be analyzed. Example of price data displayed on Fig. 1. This data depend on many uncontrollable factors such as market manipulations, news, investors' interest and probably some more. Knowing that it is impossible to make a reliable prediction of price movement in the long term (e.g. months). However short terms predictions based on technical indicators may work sometimes.

![Fig. 1. Price data example][fig1]
