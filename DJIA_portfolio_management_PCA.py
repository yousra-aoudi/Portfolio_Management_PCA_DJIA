# 1. Problem definition
"""
A primary objective of portfolio management is to allocate capital into different asset classes (equities, fixed income,
FX..) in order to maximize risk-adjusted returns. To achieve this goal, we will use PCA on a dataset of stocks.

The dataset to be used for this case study is the Dow Jones Industrial Average (DJIA) index and its respective 30 stocks.
The return data used will be from the year 2020 onwards.

I will also compare the performance of the hypothetical portfolios against a benchmark and backtest the model to
evaluate the effectiveness of the approach.
"""
# 2. Getting Started- Loading the data and python packages
# 2.1. Loading the python packages
# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Yahoo for dataReader
import yfinance as yf
yf.pdr_override()

#Import Model Packages
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import inv, eig, svd
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
# Loading data - DJI

dji_stocks = ['UNH','GS','HD','MSFT','AMGN','MCD','CAT','V','HON','JNJ','TRV','CVX','CRM','AXP','MMM','PG','AAPL','IBM',
             'BA','WMT','JPM','NKE','DIS','MRK','DOW','KO','VZ','INTC','CSCO','WBA']

start_date = '2020-1-1'
end_date = '2022-5-20'
dataset = yf.download(dji_stocks,start=start_date,end=end_date)['Adj Close']
dataset.to_csv('DJIA_close_price_pandemic_data.csv')

"""
"""
We import the dataframe containing the adjusted closing prices for all the companies in the DJIA index:
"""
dataset = read_csv('DJIA_close_price_pandemic_data.csv',index_col=0,parse_dates=True)

# 3. Exploratory Data Analysis
# 3.1. Descriptive Statistics

# Data type and description of data
set_option('display.width', 100)
print('DJI dataset \n', dataset.head(5))
print('DJI data type \n', type(dataset))

set_option('precision', 3)
print('DJI describe dataset \n', dataset.describe())

# 3.2. Data Visualization
"""
Let us take a look at the correlation. 
"""
# correlation
correlation = dataset.corr()
plt.figure(figsize=(15,15))
plt.title('Correlation Matrix')
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.savefig('DJI stocks - correlation matrix.png')
plt.show()

"""
Analysis - we can notice in the chart a significant positive correlation between the stocks.
"""
# 4. Data Preparation
# 4.1. Data Cleaning
"""
First, we check for them with the mean of the column: NAs in the rows and either drop them or fill
"""

# Checking for any null values and removing the null values
print('Null Values =',dataset.isnull().values.any())
"""
Some stocks were added to the index after our start date. To ensure proper analysis, we will drop those with more 
than 30% missing values. One stocks fit this criteria — Dow Chemicals:
"""
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
print('Missing values \n',missing_fractions.head(10))

drop_list = sorted(list(missing_fractions[missing_fractions > 0.5].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
print('dataset shape \n',dataset.shape)

"""
We end up with return data for 29 companies and an additional one for the DJIA index. Now we fill the NAs with the 
mean of the columns:
"""
# Fill the missing values with the last value available in the dataset.
dataset=dataset.fillna(method='ffill')
print('dataset after cleaning \n',dataset)

# Computing Daily Return - Linear Returns (%)
datareturns = dataset.pct_change(1)

# Remove Outliers beyong 3 standard deviation
datareturns= datareturns[datareturns.apply(lambda x :(x-x.mean()).abs()<(3*x.std()) ).all(1)]
print('data returns \n',datareturns)

# 4.2. Data transformation.
"""
In addition to handling the missing values, we also want to standardize the dataset features onto a unit scale 
(mean = 0 and variance = 1). All the variables should be on the same scale before applying PCA; otherwise, a feature 
with large values will dominate the result. We use StandardScaler in sklearn to standardize the dataset.
"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(datareturns)
rescaledDataset = pd.DataFrame(scaler.fit_transform(datareturns),columns = datareturns.columns,
                               index = datareturns.index)
# summarize transformed data
datareturns.dropna(how='any', inplace=True)
rescaledDataset.dropna(how='any', inplace=True)
print('Rescaladed dataset \n',rescaledDataset.head())

"""
Analysis - Overall, cleaning and standardizing the data is important in order to create a meaningful and reliable 
dataset to be used in dimensionality reduction without error.
"""

# Visualizing Log Returns of one of the stocks from the cleaned and standardized dataset
plt.figure(figsize=(16, 5))
plt.title("AAPL Return")
rescaledDataset.AAPL.plot()
plt.grid(True)
plt.legend()
plt.savefig("AAPL Return.png")
plt.show()

# 5. Evaluate algorithms and models
# 5.1. Train-test split.
"""
The portfolio is divided into training and test sets to perform the analysis regarding the best portfolio and to 
perform backtesting:
"""
# Dividing the dataset into training and testing sets
percentage = int(len(rescaledDataset) * 0.8)
X_train = rescaledDataset[:percentage]
X_test = rescaledDataset[percentage:]

X_train_raw = datareturns[:percentage]
X_test_raw = datareturns[percentage:]

stock_tickers = rescaledDataset.columns.values
n_tickers = len(stock_tickers)

# 5.2. Model Evaluation- Applying Principal Component Analysis - PCA
"""
As this step, we create a function to compute principal component analysis from sklearn. This function computes an 
inversed elbow chart that shows the amount of principal components and how many of them explain the variance threshold.
"""
pca = PCA()
PrincipalComponent = pca.fit(X_train)

# First Principal Component /Eigenvector
print('PCA first component \n',pca.components_[0])

# 5.2.1.Explained Variance using PCA
NumEigenvalues = 10
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
Series1 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).sort_values()*100
Series2 = pd.Series(pca.explained_variance_ratio_[:NumEigenvalues]).cumsum()*100
print('Series 2 \n',Series2)

Series1.plot.barh(ylim=(0,9), label="woohoo",title='Explained Variance Ratio by Top 10 factors',ax=axes[0])
Series2.plot(ylim=(0,100),xlim=(0,9),ax=axes[1], title='Cumulative Explained Variance by factor')
plt.savefig('Explained Variance Ratio by Top 10 factors & Cumulative Explained Variance by factor.png')
plt.show()
# explained_variance
pd.Series(np.cumsum(pca.explained_variance_ratio_)).to_frame('Explained Variance').head(NumEigenvalues).style.\
    format('{:,.2%}'.format)
"""
By the chart, the most important factor explains around 40% of the daily return variation. This dominant 
principal component is usually interpreted as the “market” factor. The plot on the right shows the cumulative explained 
variance and indicates that around ten factors explain about 73% of the variance in returns of the 29 stocks analyzed.
"""

# 5.2.2.Looking at Portfolio weights
"""
We compute several functions to determine the weights of each principle component. We then visualize a scatter-plot that 
visualizes an organized descending plot with the respective weight of every company at the current chosen principle 
component.
"""

def PCWeights():
    """
    Principal Components (PC) weights for each 29 PCs
    """
    weights = pd.DataFrame()

    for i in range(len(pca.components_)):
        weights["weights_{}".format(i)] = pca.components_[i] / sum(pca.components_[i])

    weights = weights.values.T
    return weights

weights=PCWeights()
print('weights \n',weights[0])
print('PCA first component \n',pca.components_[0])
print('weights \n',weights[0])

"""
Let us now construct five portfolios, defining the weights of each stock as each of the first five principal 
components. We then create a scatter-plot that visualizes an organized descending plot with the respec‐ tive weight of 
every company at the current chosen principal component:
"""
NumComponents = 5

topPortfolios = pd.DataFrame(pca.components_[:NumComponents], columns=dataset.columns)
print('Top portfolios \n',topPortfolios)
eigen_portfolios = topPortfolios.div(topPortfolios.sum(1), axis=0)
eigen_portfolios.index = [f'Portfolio {i}' for i in range(NumComponents)]
np.sqrt(pca.explained_variance_)
eigen_portfolios.T.plot.bar(subplots=True, layout=(int(NumComponents), 1), figsize=(14, 10), legend=False, sharey=True,
                            ylim=(-1, 1))
plt.savefig('PCA - Portfolios.png')
plt.show()

# plotting heatmap
sns.heatmap(topPortfolios)
plt.savefig('DJIA - Heatmap Top Portfolios.png')
plt.show()

"""
Traditionally, the intuition behind each principal portfolio is that it represents some sort of independent risk factor.
The manifestation of those risk factors depends on the assets in the portfolio. In this case study, the assets are all 
U.S. domestic equities. The principal portfolio with the largest variance is typically a systematic risk factor 
(i.e., “market” factor). Looking at the first principal component (Portfolio 0), we see that the weights are distributed
homogeneously across the stocks. This nearly equal weighted portfolio explains 40% of the variance in the index and is 
a fair representation of a systematic risk factor.

The rest of the eigen portfolios typically correspond to sector or industry factors. For example, Portfolio 2 assigns a
high weight to JNJ and MRK, which are stocks from the health care sector. Similarly, Portfolio 1 has high weights on 
technology and electronics companies, such AAPL, MSFT, and IBM.

When the asset universe for our portfolio is expanded to include broad, global investments, we may identify factors 
for international equity risk, interest rate risk, commodity exposure, geographic risk, and many others.
"""


# 5.2.3.Finding the Best Eigen Portfolio

# Sharpe Ratio
def sharpe_ratio(ts_returns, periods_per_year=252):
    n_years = ts_returns.shape[0] / periods_per_year
    annualized_return = np.power(np.prod(1 + ts_returns), (1 / n_years)) - 1
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol
    return annualized_return, annualized_vol, annualized_sharpe


"""
We construct a loop to compute the principal component weights for each eigen portfolio. Then it uses the Sharpe ratio 
function to look for the portfolio with the highest Sharpe ratio. Once we know which portfolio has the highest Sharpe 
ratio, we can visualize its performance against the index for comparison:
"""


def optimizedPortfolio():
    n_portfolios = len(pca.components_)
    annualized_ret = np.array([0.] * n_portfolios)
    sharpe_metric = np.array([0.] * n_portfolios)
    annualized_vol = np.array([0.] * n_portfolios)
    highest_sharpe = 0
    stock_tickers = rescaledDataset.columns.values
    n_tickers = len(stock_tickers)
    pcs = pca.components_

    for i in range(n_portfolios):
        pc_w = pcs[i] / sum(pcs[i])
        eigen_prtfi = pd.DataFrame(data ={'weights': pc_w.squeeze()*100},index = stock_tickers)
        eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True)
        eigen_prti_returns = np.dot(X_train_raw.loc[:, eigen_prtfi.index], pc_w)
        eigen_prti_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_train_raw.index)
        er, vol, sharpe = sharpe_ratio(eigen_prti_returns)
        annualized_ret[i] = er
        annualized_vol[i] = vol
        sharpe_metric[i] = sharpe

        sharpe_metric= np.nan_to_num(sharpe_metric)
    # find portfolio with the highest Sharpe ratio
    highest_sharpe = np.argmax(sharpe_metric)

    print('Eigen portfolio #%d with the highest Sharpe. Return %.2f%%, vol = %.2f%%, Sharpe = %.2f' %
          (highest_sharpe, annualized_ret[highest_sharpe]*100, annualized_vol[highest_sharpe]*100,
           sharpe_metric[highest_sharpe]))

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    ax.plot(sharpe_metric, linewidth=3)
    ax.set_title('Sharpe ratio of eigen-portfolios')
    ax.set_ylabel('Sharpe ratio')
    ax.set_xlabel('Portfolios')

    results = pd.DataFrame(data={'Return': annualized_ret, 'Vol': annualized_vol,'Sharpe': sharpe_metric})
    results.dropna(inplace=True)
    results.sort_values(by=['Sharpe'], ascending=False, inplace=True)
    print(results.head(5))
    plt.savefig('Sharpe ratio of Eigen-Portfolios.png')
    plt.show()


optimizedPortfolio()

weights = PCWeights()
portfolio = pd.DataFrame()


def plotEigen(weights, plot=False, portfolio=portfolio):
    portfolio = pd.DataFrame(data ={'weights': weights.squeeze()*100}, index = stock_tickers)
    portfolio.sort_values(by=['weights'], ascending=False, inplace=True)
    if plot:
        print('Sum of weights of current eigen-portfolio: %.2f' % np.sum(portfolio))
        portfolio.plot(title='Current Eigen-Portfolio Weights', figsize=(12, 6), xticks=range(0, len(stock_tickers), 1),
                       rot=45, linewidth=3)
        plt.savefig('Current Eigen-Portfolio Weights.png')
        plt.show()

    return portfolio

print('portfolio \n',portfolio)

# Weights are stored in arrays, where 0 is the first PC's weights.
plotEigen(weights=weights[0], plot=True)

# 5.2.4. Backtesting the eigen portfolios
"""
We will now try to backtest this algorithm on the test set, by looking at few top and bottom portfolios.
"""

def Backtest(eigen):
    """
    Plots principal components returns against real returns.
    """
    eigen_prtfi = pd.DataFrame(data={'weights': eigen.squeeze()}, index = stock_tickers)
    eigen_prtfi.sort_values(by=['weights'], ascending=False, inplace=True )

    eigen_prti_returns = np.dot(X_test_raw.loc[:, eigen_prtfi.index], eigen)
    eigen_portfolio_returns = pd.Series(eigen_prti_returns.squeeze(), index=X_test_raw.index)
    returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)

    print('Current Eigen-Portfolio:\n Return = %.2f%%\n Volatility = %.2f%%\n \ Sharpe = %.2f' %
          (returns * 100, vol * 100, sharpe))
    equal_weight_return = (X_test_raw * (1 / len(pca.components_))).sum(axis=1)
    df_plot = pd.DataFrame({'EigenPorfolio Return': eigen_portfolio_returns, 'Equal Weight Index': equal_weight_return},
                           index = X_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the equal weighted index vs. First eigen-portfolio', figsize=(12, 6),
                                 linewidth=3)
    plt.savefig('Returns of the equal weighted index vs. First eigen-portfolio.png')
    plt.show()


Backtest(eigen=weights[2])
Backtest(eigen=weights[16])
Backtest(eigen=weights[1])

"""
In this case study, we applied dimensionality reduction techniques in the context of portfolio management, 
using eigenvalues and eigenvectors from PCA to perform asset allocation.
The first eigen portfolio represented a systematic risk factor, while others exhibited sector or industry concentration.
"""