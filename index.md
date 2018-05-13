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
[fig6]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/confusion_matrices.png
[fig7]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/d_sin_medium.png
[fig8]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/q_sin_medium.png
[fig9]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/bh_medium.png
[fig10]: https://raw.githubusercontent.com/fink-stanislav/AutomaticTradingSystem/master/images/demo_profitabilities_medium.png

   * [I. Definition](#i-definition)
      * [Project Overview](#project-overview)
         * [Existing research](#existing-research)
         * [Relevance](#relevance)
         * [Datasets](#datasets)
      * [Problem Statement](#problem-statement)
         * [Problem structure](#problem-structure)
         * [Feature selection](#feature-selection)
            * [Creation of isolated environment](#creation-of-isolated-environment)
            * [Baseline and primary learning algorithm implementation](#baseline-and-primary-learning-algorithm-implementation)
            * [Learning process establishment](#learning-process-establishment)
            * [Building an application](#building-an-application)
      * [Metrics](#metrics)
         * [Profitability](#profitability)
         * [Accuracy and recall](#accuracy-and-recall)
   * [II. Analysis](#ii-analysis)
      * [Data Exploration](#data-exploration)
      * [Exploratory Visualization](#exploratory-visualization)
      * [Algorithms and Techniques](#algorithms-and-techniques)
         * [Q-Learning](#q-learning)
         * [Deep Q-Learning](#deep-q-learning)
         * [Benchmark](#benchmark)
   * [III. Methodology](#iii-methodology)
      * [Data Preprocessing](#data-preprocessing)
      * [Implementation](#implementation)
         * [Metrics](#metrics-1)
         * [Abstractions](#abstractions)
            * [Time](#time)
            * [Environments](#environments)
            * [Agents](#agents)
            * [Learners](#learners)
            * [Runners](#runners)
            * [Classifier](#classifier)
            * [Traders](#traders)
         * [Services](#services)
         * [Application](#application)
      * [Refinement](#refinement)
         * [Optimization](#optimization)
            * [Removing random exploration](#removing-random-exploration)
            * [Granting rewards](#granting-rewards)
            * [Features calculation](#features-calculation)
   * [IV. Results](#iv-results)
      * [Model Evaluation and Validation](#model-evaluation-and-validation)
      * [Justification](#justification)
      * [Demo results](#demo-results)
   * [V. Conclusion](#v-conclusion)
      * [Reflection](#reflection)
      * [Improvement](#improvement)

# I. Definition

## Project Overview

Algorithmic trading attracted many researchers in recent years. Their goal is to create software that is able to trade autonomously and profitably. Most of the works and projects are focused on traditional stock markets ([D. W. Lu](https://arxiv.org/abs/1707.07338v1), [Wang et al](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/5/5f/Dtq.pdf)) or fiat currency markets while this project is focused on cryptocurrency market.

A cryptocurrency is a digital asset designed to work as a medium of exchange using cryptography to secure the transactions, to control the creation of additional units, and to verify the transfer of assets. [Cryptocurrencies](https://en.wikipedia.org/wiki/Cryptocurrency) are classified as a subset of digital currencies and are also classified as a subset of alternative currencies and virtual currencies.

Cryptocurrency market refers to the collection of markets and exchanges where trading of cryptocurrencies and [tokens](https://en.wikipedia.org/wiki/Initial_coin_offering) takes place, either through formal exchanges or over-the-counter markets.

Cryptocurrency market is much more volatile than stock market. Volatility of stock market is summarized by [VIX](https://finance.yahoo.com/quote/%5EVIX/) index and is around 16% a year for now. Volatility of cryptocurrency market is not summarized by commonly accepted index. But still there are quite illustrative indices e.g. [SIFR Volatility Index](https://www.sifrdata.com/cryptocurrency-volatility-index/), it is around 150% for now.

Increased volatility creates more buy and sell opportunities and therefore such market is more [suitable](https://www.investopedia.com/ask/answers/010915/volatility-good-thing-or-bad-thing-investors-point-view-and-why.asp) for short-term technical analysis based trading.

Stock and fiat currencies markets differs from cryptocurrency market in terms of regulation, volatility and risk of market manipulation but the basic idea is the same. Therefore approaches that were discussed in stock market related works should also be suitable for cryptocurrency market.

### Existing research

There is a work dedicated to autonomous trading system based on Deep Q-learning. This work is quite inspiring because authors shown that Deep Q-learning produced the most significant results and their system was above baselines.

### Relevance

There are plenty of reasons why is this research relevant:
 - cryptocurrency market is quite young and there are little or no works related to its automation
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
Q-Learning and Deep Q-Learning algorithms will be implemented in this project. It is expected that Deep Q-Learning algorithm shows better performance that is why it is considered as primary while Q-Learning algorithm is considered as a baseline. Both algorithms use the same set of features and same learning process. Their core components are the only difference. Q-Learning algorithm uses table for classification and its deep counterpart uses neural network.

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

Trading system will support multiple currencies. For the purposes of the project datasets for the following currency pairs for one year will be used:

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
Figure 1. Ethereum (ETH) price in Bitcoin (BTC). Price dynamics for one year.

In this project three technical indicators will be used as features: [RSI](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi), [MFI](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi), [PPO](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:price_oscillators_ppo). The Relative Strength Index (RSI) developed by Welles Wilder, is a momentum oscillator that measures the speed and change of price movements. RSI oscillates between 0 and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30. Signals can also be generated by looking for divergences, failure swings, and centerline crossovers. RSI can also be used to identify the general trend.

The Money Flow Index (MFI) is an oscillator that uses both price and volume to measure buying and selling pressure. Created by Gene Quong and Avrum Soudack, MFI is also known as volume-weighted RSI. MFI starts with the typical price for each period. Money flow is positive when the typical price rises (buying pressure) and negative when the typical price declines (selling pressure). A ratio of positive and negative money flow is then plugged into an RSI formula to create an oscillator that moves between 0 and 100. As a momentum oscillator tied to volume, the Money Flow Index (MFI) is best suited to identify reversals and price extremes with a variety of signals. Chart displaying of RSI and MFI is shown in Fig. 2.

The Percentage Price Oscillator (PPO) is a momentum oscillator that measures the difference between two moving averages as a percentage of the larger moving average. As with its cousin, [MACD](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd), the Percentage Price Oscillator is shown with a signal line, a histogram and a centerline. Signals are generated with signal line crossovers, centerline crossovers, and divergences. There are a few differences between PPO and MACD. First, PPO readings are not subject to the price level of the asset. Second, PPO readings for different assets can be compared, even when there are large differences in the price. Chart for PPO is shown on Fig. 3

![Fig. 2. RSI and MFI][fig2]
Figure 2. RSI and MFI. Oversold zone is below, overbought zone is above. These indicators correspond to price dynamics depicted in Fig. 1.

![Fig. 3. PPO chart][fig3]
Figure 3. PPO chart. Histogram is a difference between PPO line and signal line. Histogram changes sign when signal line goes above (positive) or below (negative) PPO line.

These indicators are basic but they are very useful. There are a plenty of other indicators but their considering is out of scope of this project.

## Algorithms and Techniques

### Q-Learning

The problem is planned to be solved using two algorithms - Q-Learning and Deep Q-Learning. The former should be chosen as baseline because performance of the latter is expected to be higher. Defined problem is essentially a classification and classification agents should be trained to obtain any results. Training that uses dataset stored in advance is referred further as Batch Learning. Training that occurs in runtime and uses newly retrieved data is referred as Online Learning. Batch Learning should be run to prepare agent to work. Multiple runs with different parameters should be performed to get better results. Technique used for finding more optimal parameters for agents by means of multiple runs is called Grid Search.

Grid Search will look for the following parameters:
 - reward delay
 - alpha
 - gamma

Alpha is a learning rate. The learning rate determines to what extent newly acquired information overrides old information. A rate of 0 makes the agent learn nothing, while a rate of 1 makes the agent consider only the most recent information. In this project the learning rate will be set as a constant however learning rate is often calculated as decreasing function on each learning iteration.

Gamma is a discount rate. The discount rate determines the importance of future rewards. A factor of 0 will make the agent short-sighted by only considering current rewards, while a factor approaching 1 will make it strive for a long-term high reward.

Reward delay is a time measured in time periods which passes between taking an action and rewarding it.

A multiple instances of Grid Search should be started in order to learn different agents for different time periods. This will create agents for three different periods. So each currency pair will have three corresponding agents. The rationale behind this approach is that trading decision can be made by ensemble of agents because each of them perceives the environment with different level of detalization. Ensemble implementation will be trivial for this project (a decision provided by the most profitable agent executes) due to lack of time. But the approach looks interesting and it is definitely a matter for further research.

Environment for the project is an entity that is able to emulate changes of exchange's state over time. However the time of environment should be controllable and changeable by discrete ticks only when needed. Except changes of exchange the environment can emulate trades. Environment provides data and a space for actions for learning agents. So it will be heavily used in learning process.

One iteration of learning process will consist of the following phases:
 - calculate current state of the environment and current price of an asset
 - execute all three possible actions 'buy', 'sell', 'hold' and store them for future rewarding
 - collect rewards for actions stored on previous iteration and associate these rewards with the
 - current state by means of updating neural network or table
 - switch to next environment state
 - repeat until dataset is over

### Deep Q-Learning

Deep Q-Learning algorithm is also a Q-Learning algorithm but with neural network as action-value function approximator. It is expected to perform better than table-based version of the algorithm because it overcomes some its drawbacks.

This algorithm is able to handle non-discrete states while it is impossible for original table-based version because table would grow extremely large. Deep version is able to generalize i.e. map previously unknown states to actions mapped to similar states.

### Benchmark

To decide if project makes sense and one algorithm performs better than the other, its results should be compared to baselines. Three different baselines will be calculated for the project - cryptocurrency lending (risk-less asset), buy-and-hold strategy and Q-Learning algorithm. Deep Q-Learning results should be compared to these baselines.

Establishing the first baseline requires collecting data related to lending rates and loans demand. It should be collected by means of specially designed lending agent. The second baseline can be immediately calculated using historical data.

# III. Methodology

## Data Preprocessing

As was stated above, raw datasets from exchange cannot be used for the proposed solution. To provide input features raw data should be converted to technical indicators - RSI, MFI, PPO.

RSI and MFI can be represented by one value for each date. PPO consists of five values - exponential moving averages for 26 and 12 periods, histogram, signal line and PPO-line for each date. All five values are summarized by histogram so only histogram can be taken as a feature. Other four values used mostly for readability for humans.

Obtained indicators are essentially percentages so can be easily normalized to fall in interval from zero to one. The normalization is required for proper functioning of neural network. While Deep Q-Learning algorithm is able to use normalized features as input, table based Q-Learning requires one more preprocessing step. All features are initially real numbers that are not suitable for Q-Learning due to inappropriate growth of table. That is why features should be rounded to some degree. As rounding to the nearest integer does not make sense here rounding to the nearest 0.05 is applied.

After introducing rounding it turned out that normalized PPO value is too small to apply it. Practically PPO values do not overlap interval from -10 to 10 so normalization procedure for PPO was changed. It made PPO values 10 times larger and applicable to rounding.

## Implementation

During implementation of both algorithms features of the problem were thoroughly considered and reflected in code.

### Metrics

To calculate profitability that is main performance metric of the project `Estimator` class was introduced. Its procedure named `total_profitability` calculates profitability for a given period using states for the start of the period and the end of the period. States consist of price, base balance, and quote balance.

### Abstractions

Some abstractions were introduced to organize the code. While they have not the best names they make code more reusable and clean.

#### Time
Nearly all abstractions described below act periodically and in parallel. To demonstrate their work a decent amount of time is needed. That is why scaling of time was introduced. Number of seconds in one real second can be set in application configuration. In runtime all delays are calculated using this ratio. This approach helps to speed up the processes running in the application and see progress after reasonable amount of time. For reference see `model\_time` module. According to configuration `model\_time` module can be switched to real time scale seamlessly.

#### Environments
Two types of environments coexist in the project. The first environment residing in `environment` module is needed for learning. Its main feature is controllable state transition - it switches to the next state only if needed. When batch learning is running it executes all required learning actions and only after that switches environment to the next state. It guarantees that all states will be processed. The second environment is placed to `demo` module. It is an implementation of exchange client but with mocked order book and balances. While it provides similar functionality like emulating buys and sells, it is closer to real exchange and switches to the next state automatically after a given delay.

#### Agents
Implementations of both Q-Learning algorithms are represented as two classes in `agents` module. `QAgent` class contains a q-table in form of dictionary while `DeepQAgent` counterpart contains neural network instead. With all advantages of the neural network mentioned above, it is neither large nor complicated, see Fig. 4. It has two hidden layers of size 24 with RELU activation functions, three input neurons, and three output neurons. Curiously that larger and smaller networks performed worse.

Main differences of these classes are in methods `learn` and `choose\_action`. NN-based agent utilizes backward propagation in order to train its model while table-based one only checks if state is stored in table and updates it if it is and stores it if it is not.

Every agent has metadata, in other words information about agent itself, and associated model. In context of agent a model is trained neural network for Deep Q-Learning and filled table for Q-Learning. This information is required to be stored in order to be reusable. However agents are not aware of how and where to store it because they should not be responsible for this. This work is delegated to other entities introduced below.

#### Learners
In order to encapsulate batch learning, online learning and grid search processes in some entity, `Learning` classes were introduced and resided in `learning` module. Online and batch learnings have algorithm specific implementations just because of additional preprocessing step required for Q-Learning. They can be merged into algorithm independent implementation but it requires additional time and efforts.

Online and batch learnings are intended to train agents but they do it in different way. Online learning trains already existing agent that needs to be updated. It is expected that online learning performs one or several learning iterations, returns updated model and terminates.

Batch learning in turn creates an agent from scratch, trains it, performs roll-forward cross validation and estimates profitability of newly created agent. Afterwards it returns the agent and some metadata useful for grid search. Trading data is essentially a time series so training data should not be ahead of testing data. So usual cross-validation technique is not applicable for batch learning. As an alternative a [roll-forward cross-validation](https://robjhyndman.com/hyndsight/tscv/) was used for algorithm assessment. It prevents look-ahead bias from occurring. According to roll-forward cross-validation approach testing occurs after every example is learned. Agent's method named emulate_action is responsible for that.

Grid search also has algorithm specific implementations but they are trivial and needed only for convenient way of creating batch learnings. However they also could be merged after some code cleanup. Grid search tries to find an agent with maximal profitability by means of batch learning. Also it is able to estimate its own progress of doing that. In the application grid search is created for every currency pair with data snapshot for one year. Being rather trivial grid search implementation still has an open question about gamma parameter. This parameter is quite contradictory for this particular project because its values are hard to interpret. It turned out to be difficult to correlate results and these values. Possibly this parameter should be replaced by reward delay parameter but additional tests are needed to prove that.

#### Runners
Implemented Q-Learning algorithms require additional maintenance code. That maintenance code should provide the following functionality:

 - store and load metadata
 - store and load models
 - monitor runtime state
 - call learning procedure when needed

For encapsulating this functionality `Runners` were introduced. There are runners for online learning and for grid search. Runners for online learning call learning procedure periodically while grid search runners call it only once. Internally runners for online learning (run runners) encapsulate a thread that executes learning procedure with given interval. Interval is equal to time period assigned to corresponding agent. Also run runners have a method that returns an action for a given state. Each agent has its own instance of runner.

#### Classifier
To unite agents and their runners and make them work as an unified system one more entity is needed. This entity serves as an interface for all classifications made by agents that is why it is called `Classifier`. Its responsibility includes the following:

 - starting and stopping required runners on demand
 - checking statuses of runners 
 - requesting classifications for current state of environment

Only one instance of this class is required for proper work of the application. Instance of classifier is utilized by Agents Service described in Services section below.

#### Traders
Traders are the users of classifications provided by `Classifier`. Instance of Trader class from trading module is responsible for the following:

 - placing and canceling orders
 - requesting the latest classifications (or actions)
 - checking orders execution status periodically and replacing them if needed

Just as runners and classifier traders encapsulate a thread that runs periodically. The thread is responsible for executing a procedure of placing or replacing orders according to retrieved action suggestion and order's execution status.

Number of traders is not fixed, however for demonstration purposes only ten traders will be used. Five will use Q-Learning and another five will use Deep Q-Learning.

To manage traders `TradersManager` class was introduced. It is able to request for status, and also start and stop traders. It stores all running traders inside a dictionary. `TradersManager` is an interface for all traders and it is used by traders service.

### Services

Agents and traders are essentially loosely coupled entities because they do not need each others' code. The only thing a traders needs from agents is a proposed action, while agents do not care of traders at all. This fact is a motivation for isolating these entities from each other but with possibility to interact. A widely accepted way of doing this is creating web services.

Two web services were created - Agents Service and Traders Service. Agents Service analyzes a current state of exchange and suggests actions corresponding to that state. Traders Service requests these suggestions and performs trades.

All interactions are made by means of HTTP requests and JSON. [Web.py](http://webpy.org/) framework used for all boilerplate code.

### Application

In order to use implemented software effectively and conveniently a GUI is required. Constructing a website is quick and flexible way to display required information and provide control over desired systems. Therefore a website with three pages was designed.

Overview page shows profitability of all traders and helps to visually compare them to each other.

Agents and Traders page shows names of all started agents and traders and short related metadata. From both pages links to Trader Details page are available.

Trader Details page displays price chart with all trades made by trader and chart for price of trader's holdings. This page is parameterized by trader name. Current balances and order book are also shown. Order book however empties too quickly because of time scaling of demo. It is matter of seconds to execute an order so one who looks at demo could easily miss it.

All pages use services described above to retrieve data. For creating the application [Flask](http://flask.pocoo.org) framework was selected for its minimalistic simplicity.

For the purposes of this report the site is set up to show a demo version of working software. This approach intends to prove that problem is solved by showing all components working together with their performance measured by profitability and prove it in reasonable time. Currently the speed of demo is not very large. It is 200 times faster than real world and it is hard to speed it up more. Learning, checking, storing and other processes take time and need to be even more optimized.

## Refinement

### Optimization

Minor algorithm optimizations were applied during implementation. They allowed to speed up the execution in several times.

#### Removing random exploration
Originally Q-Learning algorithm includes random exploration but as the number of possible actions is low it does not make sense to use it. So learning procedure executes all three actions independently and checks their results after reward delay is up. This helps to speed up learning process and improve results.

#### Granting rewards
Initially rewards granting process was different. Three actions were executed and learner switched to N states forth to estimate actions' results and then switched back to reward them. It was complicated and not optimal process, possibly introducing look-ahead bias. "Possibly" because classification performance was poor for some reason. Now switches back and forth are removed and environment only switches forth one time during one learning iteration.

#### Features calculation
It turned out that operations on series are faster than operations on dataframes. Module named `indicators` is a good illustration of that. Methods with postfix `df` are nearly four times slower than methods without postfix.

# IV. Results

## Model Evaluation and Validation

Profitability is used as main performance metric of the project. Both Q-Learning algorithms demonstrated good profitability. Their profitability in percents is shown in Fig. 5.

![Fig. 5. Profitabilities][fig5]
Figure 5. Profitabilities of different algorithms on given pairs and periods.

As can be seen on the graph most of the times profitability of Deep Q-Learning is higher. But it grows on increase of the number of trades (the lower the period the higher is the number of trades).

Table 2. Average profitabilities of algorithms by period

|Period|Profitability of Deep Q-Learning, %|Profitability of Q-Learning, %|Difference in percents|
|-|-|-|-|
|7200|17343|11248|54.18|
|14400|7864|7541|4.27|
|86400|767|780|-1.57|

As can be seen from Table 2 both algorithms demonstrate the best performance for period 7200. It is reasonably to guess that shorter period would give even better performance but it is not the case. Profitability starts to degrade with shorter periods because data becomes too noisy. Overall profit for the case when funds are equally allocated between currencies is 86718% for Deep Q-Learning and 56244% for Q-Learning for training period.

Statistically the advantage of Deep Q-Learning is also confirmed. It shows higher accuracy and recall. Both algorithms show higher accuracy and recall comparing to random actions, see Table 3.

Table 3. Statistical measures of algorithms’ performance

|Metric|Deep Q-Learning|Q-Learning|Random actions|
|-|-|-|-|
|Accuracy|0.670|0.637|0.482|
|Recall|0.765|0.695|0.676|

Confusion matrices are displayed in Fig. 6. Matrices for both Q-Learning algorithms are nearly the same, however results for deep version are little bit better.

![Fig. 6. Confusion matrices][fig6]
Figure 6. Confusion matrices for Deep Q-Learning, Q-Learning, and random actions. Period is 7200. Values are denoted in percents.

Another proof of algorithms' comprehension is applying them to synthetic data. Dataset which prices form sinusoidal curve was generated and both algorithms were trained and tested on it. Both algorithms grasp knowledge regarding price changes correctly and quickly. However Deep Q-Learning algorithm provides noisier result - it commits multiple trades when it is not really needed. See some consequent buys and sells in minimum and maximum in Fig. 7. Q-Learning in turn makes trades very smoothly, see Fig. 8. Nevertheless profitabilities for both algorithms are almost the same and profitability of deep version is ca. 1% higher.

![Fig. 7. Results on synthetic data for Q-Learning][fig7]
Figure 7. Trades of Deep Q-Learning on synthetic (sinusoid) price data.

![Fig. 8. Results on synthetic data for Deep Q-Learning][fig8]
Figure 8. Trades of Q-Learning on synthetic (sinusoid) price data.

Both algorithms are shown to be quite effective and able to gain profits. Deep Q-Learning algorithm was expected to have higher performance and it is shown that it really has. However this algorithm has large advantage only on big amount of trades, on low amount of trades its profitability is the same or just a little bit bigger.

## Justification

Deep Q-Learning algorithm performs better than Q-Learning. However it also should be compared to other baselines. Buy and hold strategy is the simplest baseline. To emulate this strategy all relevant currencies should be "bought" at the start of dataset's time frame. For every currency 0.01 of Bitcoin was allocated. As shown on Fig. 9 price bounced up and down a lot but holdings' increase is comparatively small. Initial holdings price was 0.05 Bitcoin. Final price is 0.143. Therefore buy and hold strategy could bring 285% of profit in a year.

![Fig. 9. Buy and hold][fig9]
Figure 9. Sum of balances of five relevant currencies converted to Bitcoin over time.

Another baseline is lending. It is calculated using data from exchange. Loan rates are not available just as price data, that is why it had to be collected. Also publishing these rates does not make much sense because rate is bound to period and the loan can finish before single period ends. So a client for exchange was written specifically for lending Bitcoin. It was able to collect some statistics.

The lending agent executed more than 400 lending orders in two months. For two months it was able to make 0.26% of profit so it should bring ca. 1.57% per year.

Comparing profitabilities of baselines to profitability of Deep Q-Learning algorithm has shown that its performance is good and it definitely outperforms all baselines and so it is worth implementing and using.

## Demo results

Results shown in demo also confirm better performance of Deep Q-Learning. To check all demonstration period several hours are required but to see first results an half an hour should be enough. Final profitabilities of traders depicted on Fig. 10.

![Fig. 10. Profitabilities][fig10]
Figure 10. Profitabilities of Q-Learning and Deep Q-Learning traders on the end of demo period

As can be seen on Fig. 10 profitabilities are rather high but lower than profitabilities for training period. It is because of different length of periods. Training period is one year while demo period is three months.

# V. Conclusion

## Reflection

The solution for the stated problem is quite complex. This complexity is created by mathematical and engineering parts of the project. Mathematical part consists of Q-Learning algorithms and features calculation. Engineering part is wiring everything together and providing convenient access to algorithms. These parts are essential for composing working application for automation of trading but Q-Learning algorithms are nevertheless the core of the project.

Q-Learning algorithm is derived from Smartcab project. During implementation of the project the algorithm has not been changed a lot because it was literally ready to apply. A learning process in turn was a challenge. Associating rewards with the current state basing on previous actions seemed counterintuitive at the beginning. Also initial implementations were extremely slow. Deep Q-Learning uses the same mechanisms for learning. That is why introducing this algorithm was rapid. Choosing neural network architecture also did not take long. Tests had quickly shown that networks with too many layers and too many neurons in layer did not improve performance. The best results were obtained with smaller networks with no more than two hidden layers.

Achieving good results is impossible without well-handled data. Automation of trading requires properly selected features to make decisions. There is a [project](https://arxiv.org/abs/1707.07338v1) that uses raw price data for some period of time to learn. From the one hand it makes perfect sense because price is very important source of data for predicting trends. From the other hand this approach ignores volume, which also is very important, and requires hundreds of inputs. Huge inputs automatically block the possibility of usage of Q-Learning algorithm leaving only Deep Q-Learning. This project uses alternative to raw price data - technical indicators. They make inputs tiny, do not ignore volumes and preserve history in a special way. Looking at results this approach seems to be rather effective.

Calculating features and reacting to changing market state requires the most recent data. Requesting and fetching new data, storing it, dropping obsolete - all these processes are needed for proper application's functioning. The application, in turn, uses data to train models on the fly and periodically relearn them completely. Making a lot of parallel tasks work together was another challenge of this project. To achieve that, additional abstractions were introduced alongside with algorithms. It saved project's code from being monolithic and non-reusable, however introduced some examples of bad naming.

Despite of all challenges and complications results of the project are way above expectations. Profitability of more than 1000% is really huge number in terms of trading. However this result is obtained on demo environment without placing real orders. Real performance will also depend on ability of trader to select reasonable price to quick and profitable execution of order. That is why profitability on real exchange is expected to be lower.

## Improvement

Despite of results being quite good they are still far from being perfect and there are a lot of improvements that can be applied to the project. The most obvious thing to improve is features set. Deep Q-Learning algorithm can handle more features. However there is no need to add a lot them. Some indicators that bring new information to the algorithm could possibly improve performance. A drawback of adding new features is a possible model change. Different architecture of neural network could work better with new features but it still has to be found.

Another quite obvious improvement is approach to grid search. For now grid search is planned to be executed once a month with strictly limited set of parameters to check. It seems to make sense to run grid search in two or more rounds with different sets of parameters. After first round is over a model should be updated and continue working as it was. But grid search could be restarted with a set of parameters closer to optimal values found. Another approach is to run grid search on shorter time interval and possibly more frequently. This approach could possibly lead to worse classification performance but it is still worth trying.

One more promising change is improving of ensembles. As was stated above current implementation of ensembles is trivial. It could be changed in different ways. For example a decision can be made using weighted results of all members of ensemble. Profit per action could be taken as weight.

All possible changes described above are related to classification abilities of the application. But it is not the only area to improve. Technical features are also can be improved.

Further development and extension of the application will require scalability issues to be resolved. Application should be split to even more services to have possibility to run them in parallel and on different virtual machines in the cloud. Fortunately the nature of the application is essentially parallel and this issue is rather simple.

Another possible improvement is speed. Some parts of the application are still slow. Grid search and learning procedures should be carefully investigated for bottlenecks. Also CUDA can help to speed up but this statement should also be proven.

The most tedious improvement of the project is error handling. For now it is poor and not very helpful but it could be implemented much better.

Overall the result of the project is better than expected but there is still much to improve.
