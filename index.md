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
To make trading decisions an automatic trading system should acquire the most recent data of currency prices and volumes. A major exchanges provide APIs that make this data freely available at any time. For example to retrieve data of prices and volumes of Bitcoin to Etherium trades on Poloniex for January 2018 only desired period (e.g. 1 day, 4 hours etc.) and timestamps for start and end of the month are required 8 . Data is returned in JSON and contains open, close, high and low prices for the period and traded volumes. This data is sufficient for calculating all indicators utilized for decision making.

## Problem Statement
The problem to solve is to automate trading process by means of automated trading system.

Trading is the process of buying and selling cryptocurrency on specialized exchange over the Internet. There are a lot of exchanges 9 but for the sake of clarity and simplicity the project will use Poloniex 10 and its API.

The purpose of trading is to increase total price of assets hold by trader. The price can be nominated in fiat currency or in any cryptocurrency of choice. Trader is the person who commits buys and sells of digital assets or handles automated trading system that commits these actions.

Automated trading system should be able to recognize buy and sell signals and recognize situations when to hold. Signals to buy, sell or hold should be recognized by means of machine learning. To recognize these signals machine learning part of the system should be able to infer state of the market by processing input features. Features are produced by means of technical analysis and technical indicators in particular. Technical analysis is an analysis methodology for forecasting the direction of prices through the study of past market data, primarily price and volume 11 . In technical analysis, a technical indicator is a mathematical calculation based on historic price and volume that aims to forecast financial market direction 12 .

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
As it was discussed in Capstone Proposal Q-Learning and Deep Q-Learning algorithms will be implemented in this project. It is expected that Deep Q-Learning algorithm shows better performance that is why it is considered as primary while Q-Learning algorithm is considered as a baseline. Both algorithms use the same set of features and same learning process. Their core
components are the only difference. Q-Learning algorithm uses table for classification and its deep counterpart uses neural network.

#### Learning process establishment
Learning process is the sequence of actions dedicated to train agents. Learning process for both types of agents of this project should be the same. Initially agents are trained using data stored in advance. Running agents are trained periodically after some specified delay in order to adapt to changing environment. Full retraining with dropping of trained models also should occur once in a while to prevent agents to accumulate obsolete knowledge.

#### Building an application
Algorithms and all code for their maintenance should be wrapped up in web-application in order to provide visual representation of their work and state.

## Metrics

### Profitability

Profitability is a price difference between initial holdings of trader and his/her holdings at a given moment after some trades were executed. Profitability should be denoted in percents.

Price of holdings for this particular project should be nominated in Bitcoin (BTC). It is done because a market the system is trying to beat is nominated in BTC. Also Bitcoin has many years' uptrend and is the most widely acceptable and the most recognized cryptocurrency. System's profitability should be calculated as follows:

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/fink-stanislav/AutomaticTradingSystem/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/fink-stanislav/AutomaticTradingSystem/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
