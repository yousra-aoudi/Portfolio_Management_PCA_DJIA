import bs4 as bs
import requests
import tiingoconnect
import pandas as pd
import datetime as dt
import os

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web


def S_And_P_tickers():
    '''
    Parses Slickcharts Webpage to obtain the tickers for each current listed stock in the S&P.
    '''

    resp = requests.get('https://money.cnn.com/data/sandp/')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wsod_dataTable wsod_dataTableBig'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker.split('\xa0', 1)[0]
        tickers.append(ticker)

    return tickers


# S&P_tickers()


def API_data():
    """
    Uses Tiingo API to access all the historical data for the current stocks listed in the S&P.
    """
    S_And_P = '.DJI'
    tickers = S_And_P_tickers()
    tickers.append(DJIA)

    if not os.path.exists ( 'stocks_S_And_P' ):
        os.makedirs('stocks_S_And_P' )

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now()

    for ticker in tickers:
        print (ticker)
        if not os.path.exists ( 'stocks_S_And_P/{}.csv'.format ( ticker )):
            df = tiingoconnect.DataReader ( ticker, start, end )
            df.to_csv ( 'stocks_DJIA/{}.csv'.format ( ticker ) )
        else:
            print ( 'Information already acquired for {}'.format ( ticker ) )


# API_data()

def compile_data():
    '''
    Creates a dataframe with the compiled adjusted closing for all the stocks in the S&P.
    '''

    tickers = S_And_P_tickers()

    main_df = pd.DataFrame ()

    for count, ticker in enumerate ( tickers ):
        df = pd.read_csv('stocks_S_And_P/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)

        df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash',
                   'splitFactor'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join ( df, how='outer' )

        if count % 10 == 0:
            print ( count )

    print ( main_df.head () )
    main_df.to_csv ( 'DJIA_adjcloses.csv' )

# compile_data()