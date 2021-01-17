# awesome-quant
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of insanely awesome libraries, packages and resources for Quants (Quantitative Finance)

## Languages

- [Python](#python)
- [Julia](#julia)
- [Haskell](#haskell)
- [Scala](#scala)
- [CSharp](#csharp)
- [Frameworks](#frameworks) - frameworks that support different languages
- [Reproducing Works](#reproducing-works) - repositories that reproduce books and papers results or implement examples

## Python

### Tools blog

- https://www.activestate.com/blog/top-10-python-packages-for-finance-and-financial-modeling/

### Numerical Libraries & Data Structures

- [numpy](https://www.numpy.org) - NumPy is the fundamental package for scientific computing with Python.
- [scipy](https://www.scipy.org) - SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [pandas](https://pandas.pydata.org) - pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
- [quantdsl](https://github.com/johnbywater/quantdsl) - Domain specific language for quantitative analytics in finance and trading.
- [statistics](https://docs.python.org/3/library/statistics.html) - Builtin Python library for all basic statistical calculations.
- [sympy](https://www.sympy.org/) - SymPy is a Python library for symbolic mathematics.
- [pymc3](https://docs.pymc.io/) - Probabilistic Programming in Python: Bayesian Modeling and Probabilistic Machine Learning with Theano.

### Financial Instruments and Pricing

- [PyQL](https://github.com/enthought/pyql) - QuantLib's Python port.
- [pyfin](https://github.com/opendoor-labs/pyfin) - Basic options pricing in Python. [ARCHIVED]
- [vollib](https://github.com/vollib/vollib) - vollib is a python library for calculating option prices, implied volatility and greeks.
- [QuantPy](https://github.com/jsmidt/QuantPy) - A framework for quantitative finance In python.
- [Finance-Python](https://github.com/alpha-miner/Finance-Python) - Python tools for Finance.
- [ffn](https://github.com/pmorissette/ffn) - A financial function library for Python.
- [pynance](https://pynance.net) - PyNance is open-source software for retrieving, analysing and visualizing data from stock and derivatives markets.
- [tia](https://github.com/bpsmith/tia) - Toolkit for integration and analysis.
- [hasura/base-python-dash](https://platform.hasura.io/hub/projects/hasura/base-python-dash) - Hasura quickstart to deploy Dash framework. Written on top of Flask, Plotly.js, and React.js, Dash is ideal for building data visualization apps with highly custom user interfaces in pure Python.
- [hasura/base-python-bokeh](https://platform.hasura.io/hub/projects/hasura/base-python-bokeh) - Hasura quickstart to visualize data with bokeh library.
- [pysabr](https://github.com/ynouri/pysabr) - SABR model Python implementation.
- [FinancePy](https://github.com/domokane/FinancePy) - A Python Finance Library that focuses on the pricing and risk-management of Financial Derivatives, including fixed-income, equity, FX and credit derivatives.
- [FinancePy-Examples](https://github.com/domokane/FinancePy-Examples) - Examples of how to use FinancePy
- [gs-quant](https://github.com/goldmansachs/gs-quant) - Python toolkit for quantitative finance

### Indicators
- [pandas_talib](https://github.com/femtotrader/pandas_talib) - A Python Pandas implementation of technical analysis indicators.
- [finta](https://github.com/peerchemist/finta) - Common financial technical analysis indicators implemented in Pandas.
- [Tulipy](https://github.com/cirla/tulipy) - Financial Technical Analysis Indicator Library (Python bindings for [tulipindicators]( https://github.com/TulipCharts/tulipindicators))

### Trading & Backtesting

- [bt](https://github.com/pmorissette/bt) - Flexible Backtesting for Python.
- [backtrader](https://github.com/backtrader/backtrader) - Python Backtesting library for trading strategies.
- [tradingWithPython](https://pypi.org/project/tradingWithPython/) - A collection of functions and classes for Quantitative trading.
- [Pandas TA](https://github.com/twopirllc/pandas-ta) - Pandas TA is an easy to use Python 3 Pandas Extension with 115+ Indicators. Easily build Custom Strategies.
- [ta](https://github.com/bukosabino/ta) - Technical Analysis Library using Pandas (Python)
- [finmarketpy](https://github.com/cuemacro/finmarketpy) - Python library for backtesting trading strategies and analyzing financial markets.
- [pylivetrader](https://github.com/alpacahq/pylivetrader) - zipline-compatible live trading library.
- [pipeline-live](https://github.com/alpacahq/pipeline-live) - zipline's pipeline capability with IEX for live trading.
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) - Financial portfolio optimisation in python, including classical efficient frontier and advanced methods.
- [riskparity.py](https://github.com/dppalomar/riskparity.py) - fast and scalable design of risk parity portfolios with TensorFlow 2.0
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) - Implementations regarding "Advances in Financial Machine Learning" by Marcos Lopez de Prado. (Feature Engineering, Financial Data Structures, Meta-Labeling)
- [pyqstrat](https://github.com/abbass2/pyqstrat) - A fast, extensible, transparent python library for backtesting quantitative strategies.
- [aat](https://github.com/timkpaine/aat) - Async Algorithmic Trading Engine
- [Backtesting.py](https://kernc.github.io/backtesting.py/) - Backtest trading strategies in Python
- [catalyst](https://github.com/enigmampc/catalyst) - An Algorithmic Trading Library for Crypto-Assets in Python
- [quantstats](https://github.com/ranaroussi/quantstats) - Portfolio analytics for quants, written in Python
- [qtpylib](https://github.com/ranaroussi/qtpylib) - QTPyLib, Pythonic Algorithmic Trading <http://qtpylib.io> 
- **[freqtrade](https://github.com/freqtrade/freqtrade) - Free, open source crypto trading bot**
- [algorithmic-trading-with-python](https://github.com/chrisconlan/algorithmic-trading-with-python) - Free `pandas` and `scikit-learn` resources for trading simulation, backtesting, and machine learning on financial data.
- [DeepDow](https://github.com/jankrepl/deepdow) - Portfolio optimization with deep learning

### Risk Analysis

- [pyfolio](https://github.com/quantopian/pyfolio) - Portfolio and risk analytics in Python.
- [empyrical](https://github.com/quantopian/empyrical) - Common financial risk and performance metrics.
- [fecon235](https://github.com/rsvp/fecon235) - Computational tools for financial economics include: Gaussian Mixture model of leptokurtotic risk, adaptive Boltzmann portfolios.
- [qfrm](https://pypi.org/project/qfrm/) - Quantitative Financial Risk Management: awesome OOP tools for measuring, managing and visualizing risk of financial instruments and portfolios.
- [visualize-wealth](https://github.com/benjaminmgross/visualize-wealth) - Portfolio construction and quantitative analysis.
- [VisualPortfolio](https://github.com/wegamekinglc/VisualPortfolio) - This tool is used to visualize the perfomance of a portfolio.

### Factor Analysis

- [alphalens](https://github.com/quantopian/alphalens) - Performance analysis of predictive alpha factors.
- [Spectre](https://github.com/Heerozh/spectre) - GPU-accelerated Factors analysis library and Backtester

### Time Series

- [ARCH](https://github.com/bashtage/arch) - ARCH models in Python.
- [statsmodels](http://statsmodels.sourceforge.net) - Python module that allows users to explore data, estimate statistical models, and perform statistical tests.
- [dynts](https://github.com/quantmind/dynts) - Python package for timeseries analysis and manipulation.
- [PyFlux](https://github.com/RJT1990/pyflux) - Python library for timeseries modelling and inference (frequentist and Bayesian) on models.
- [tsfresh](https://github.com/blue-yonder/tsfresh) - Automatic extraction of relevant features from time series.
- [hasura/quandl-metabase](https://platform.hasura.io/hub/projects/anirudhm/quandl-metabase-time-series) - Hasura quickstart to visualize Quandl's timeseries datasets with Metabase.

### Calendars

- [trading_calendars](https://github.com/quantopian/trading_calendars) - Stock Exchange Trading Calendars.
- [bizdays](https://github.com/wilsonfreitas/python-bizdays) - Business days calculations and utilities.
- [pandas_market_calendars](https://github.com/rsheftel/pandas_market_calendars) - Exchange calendars to use with pandas for trading applications.

### Data Sources

- [findatapy](https://github.com/cuemacro/findatapy) - Python library to download market data via Bloomberg, Quandl, Yahoo etc.
- [googlefinance](https://github.com/hongtaocai/googlefinance) - Python module to get real-time stock data from Google Finance API.
- [yahoo-finance](https://github.com/lukaszbanasiak/yahoo-finance) - Python module to get stock data from Yahoo! Finance.
- [pandas-datareader](https://github.com/pydata/pandas-datareader) - Python module to get data from various sources (Google Finance, Yahoo Finance, FRED, OECD, Fama/French, World Bank, Eurostat...) into Pandas datastructures such as DataFrame, Panel with a caching mechanism.
- [pandas-finance](https://github.com/davidastephens/pandas-finance) - High level API for access to and analysis of financial data.
- [pyhoofinance](https://github.com/innes213/pyhoofinance) - Rapidly queries Yahoo Finance for multiple tickers and returns typed data for analysis.
- [yfinanceapi](https://github.com/Karthik005/yfinanceapi) - Finance API for Python.
- [yql-finance](https://github.com/slawek87/yql-finance) - yql-finance is simple and fast. API returns stock closing prices for current period of time and current stock ticker (i.e. APPL, GOOGL).
- [ystockquote](https://github.com/cgoldberg/ystockquote) - Retrieve stock quote data from Yahoo Finance.
- [wallstreet](https://github.com/mcdallas/wallstreet) - Real time stock and option data.
- [stock_extractor](https://github.com/ZachLiuGIS/stock_extractor) - General Purpose Stock Extractors from Online Resources.
- [Stockex](https://github.com/cttn/Stockex) - Python wrapper for Yahoo! Finance API.
- [finsymbols](https://github.com/skillachie/finsymbols) - Obtains stock symbols and relating information for SP500, AMEX, NYSE, and NASDAQ.
- [FRB](https://github.com/avelkoski/FRB) - Python Client for FRED® API.
- [inquisitor](https://github.com/econdb/inquisitor) - Python Interface to Econdb.com API.
- [yfi](https://github.com/nickelkr/yfi) - Yahoo! YQL library.
- [chinesestockapi](https://pypi.org/project/chinesestockapi/) - Python API to get Chinese stock price.
- [exchange](https://github.com/akarat/exchange) - Get current exchange rate.
- [ticks](https://github.com/jamescnowell/ticks) - Simple command line tool to get stock ticker data.
- [pybbg](https://github.com/bpsmith/pybbg) - Python interface to Bloomberg COM APIs.
- [ccy](https://github.com/lsbardel/ccy) - Python module for currencies.
- [tushare](https://pypi.org/project/tushare/) - A utility for crawling historical and Real-time Quotes data of China stocks.
- [jsm](https://pypi.org/project/jsm/) - Get the japanese stock market data.
- [cn_stock_src](https://github.com/jealous/cn_stock_src) - Utility for retrieving basic China stock data from different sources.
- [coinmarketcap](https://github.com/barnumbirr/coinmarketcap) - Python API for coinmarketcap.
- [after-hours](https://github.com/datawrestler/after-hours) - Obtain pre market and after hours stock prices for a given symbol.
- [bronto-python](https://pypi.org/project/bronto-python/) - Bronto API Integration for Python.
- [pytdx](https://github.com/rainx/pytdx) - Python Interface for retrieving chinese stock realtime quote data from TongDaXin Nodes.
- [pdblp](https://github.com/matthewgilbert/pdblp) - A simple interface to integrate pandas and the Bloomberg Open API.
- [tiingo](https://github.com/hydrosquall/tiingo-python) - Python interface for daily composite prices/OHLC/Volume + Real-time News Feeds, powered by the Tiingo Data Platform.
- [iexfinance](https://github.com/addisonlynch/iexfinance) - Python Interface for retrieving real-time and historical prices and equities data from The Investor's Exchange.
- [pyEX](https://github.com/timkpaine/pyEX) - Python interface to IEX with emphasis on pandas, support for streaming data, premium data, points data (economic, rates, commodities), and technical indicators.
- [alpaca-trade-api](https://github.com/alpacahq/alpaca-trade-api-python) - Python interface for retrieving real-time and historical prices from Alpaca API as well as trade execution.
- [metatrader5](https://pypi.org/project/MetaTrader5/) - API Connector to MetaTrader 5 Terminal
- [akshare](https://github.com/jindaxiang/akshare) - AkShare is an elegant and simple financial data interface library for Python, built for human beings! <https://akshare.readthedocs.io>
- [yahooquery](https://github.com/dpguthrie/yahooquery) - Python interface for retrieving data through unofficial Yahoo Finance API.
- [investpy](https://github.com/alvarobartt/investpy) - Financial Data Extraction from Investing.com with Python! <https://investpy.readthedocs.io/>
- [yliveticker](https://github.com/yahoofinancelive/yliveticker) - Live stream of market data from Yahoo Finance websocket.
- [bbgbridge](https://github.com/ran404/bbgbridge) - Easy to use Bloomberg Desktop API wrapper for Python.

### Excel Integration

- [xlwings](https://www.xlwings.org/) - Make Excel fly with Python.
- [openpyxl](https://openpyxl.readthedocs.io/en/latest/) - Read/Write Excel 2007 xlsx/xlsm files.
- [xlrd](https://github.com/python-excel/xlrd) - Library for developers to extract data from Microsoft Excel spreadsheet files.
- [xlsxwriter](https://xlsxwriter.readthedocs.io/) - Write files in the Excel 2007+ XLSX file format.
- [xlwt](https://github.com/python-excel/xlwt) - Library to create spreadsheet files compatible with MS Excel 97/2000/XP/2003 XLS files, on any platform.
- [DataNitro](https://datanitro.com/) - DataNitro also offers full-featured Python-Excel integration, including UDFs. Trial downloads are available, but users must purchase a license.
- [xlloop](http://xlloop.sourceforge.net) - XLLoop is an open source framework for implementing Excel user-defined functions (UDFs) on a centralised server (a function server).
- [expy](http://www.bnikolic.co.uk/expy/expy.html) - The ExPy add-in allows easy use of Python directly from within an Microsoft Excel spreadsheet, both to execute arbitrary code and to define new Excel functions.
- [pyxll](https://www.pyxll.com) - PyXLL is an Excel add-in that enables you to extend Excel using nothing but Python code.

### Visualization

- [D-Tale](https://github.com/man-group/dtale) - Visualizer for pandas dataframes and xarray datasets.
- [mplfinance](https://github.com/matplotlib/mplfinance) - matplotlib utilities for the visualization, and visual analysis, of financial data.


## Julia

- [QuantLib.jl](https://github.com/pazzo83/QuantLib.jl) - Quantlib implementation in pure Julia.
- [FinancialMarkets.jl](https://github.com/imanuelcostigan/FinancialMarkets.jl) - Describe and model financial markets objects using Julia.
- [Ito.jl](https://github.com/aviks/Ito.jl) - A Julia package for quantitative finance.
- [TALib.jl](https://github.com/femtotrader/TALib.jl) - A Julia wrapper for TA-Lib.
- [Miletus.jl](https://juliacomputing.com/docs/miletus/index.html) - A financial contract definition, modeling language, and valuation framework.
- [Temporal.jl](https://github.com/dysonance/Temporal.jl) - Flexible and efficient time series class & methods.
- [Indicators.jl](https://github.com/dysonance/Indicators.jl) - Financial market technical analysis & indicators on top of Temporal.
- [Strategems.jl](https://github.com/dysonance/Strategems.jl) - Quantitative systematic trading strategy development and backtesting.
- [TimeSeries.jl](https://github.com/JuliaStats/TimeSeries.jl) - Time series toolkit for Julia.
- [MarketTechnicals.jl](https://github.com/JuliaQuant/MarketTechnicals.jl) - Technical analysis of financial time series on top of TimeSeries.
- [MarketData.jl](https://github.com/JuliaQuant/MarketData.jl) - Time series market data.
- [TimeFrames.jl](https://github.com/femtotrader/TimeFrames.jl) - A Julia library that defines TimeFrame (essentially for resampling TimeSeries).


## Haskell

- [quantfin](https://github.com/boundedvariation/quantfin) - quant finance in pure haskell.
- [hqfl](https://github.com/co-category/hqfl) - Haskell Quantitative Finance Library.

## Scala

- [QuantScale](https://github.com/choucrifahed/quantscale) - Scala Quantitative Finance Library.
- [Scala Quant](https://github.com/frankcash/Scala-Quant) Scala library for working with stock data from IFTTT recipes or Google Finance.

## Frameworks

- [QuantLib](https://www.quantlib.org) - The QuantLib project is aimed at providing a comprehensive software framework for quantitative finance.
	- [JQuantLib](http://www.jquantlib.org) - Java port.
	- [RQuantLib](http://dirk.eddelbuettel.com/code/rquantlib.html) - R port.
	- [QuantLibAddin](https://www.quantlib.org/quantlibaddin/) - Excel support.
	- [QuantLibXL](https://www.quantlib.org/quantlibxl/) - Excel support.
	- [QLNet](https://github.com/amaggiulli/qlnet) - .Net port.
	- [PyQL](https://github.com/enthought/pyql) - Python port.
	- [QuantLib.jl](https://github.com/pazzo83/QuantLib.jl) - Julia port.
- [TA-Lib](https://ta-lib.org) - perform technical analysis of financial market data.

## CSharp

- [QuantConnect](https://github.com/QuantConnect/Lean) - Lean Engine is an open-source fully managed C# algorithmic trading engine built for desktop and cloud usage.

## Reproducing Works

- [Derman Papers](https://github.com/MarcosCarreira/DermanPapers) - Notebooks that replicate original quantitative finance papers from Emanuel Derman.
- [volatility-trading](https://github.com/jasonstrimpel/volatility-trading) - A complete set of volatility estimators based on Euan Sinclair's Volatility Trading.
- [quant](https://github.com/paulperry/quant) - Quantitative Finance and Algorithmic Trading exhaust; mostly ipython notebooks based on Quantopian, Zipline, or Pandas.
- [fecon235](https://github.com/rsvp/fecon235) - Open source project for software tools in financial economics. Many jupyter notebook to verify theoretical ideas and practical methods interactively.
- [Quantitative-Notebooks](https://github.com/LongOnly/Quantitative-Notebooks) - Educational notebooks on quantitative finance, algorithmic trading, financial modelling and investment strategy
