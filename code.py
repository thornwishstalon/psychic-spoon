import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import os
import re
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

data_prefix = "data/"


# HELPER   ###################################
def create_year_dict(range, type, postfix=None):
    year_dict = {}
    for i in range:
        key = str(i)
        if postfix is not None:
            key = str(i) + postfix.format(i)

        year_dict[key] = type

    return year_dict


def load_country_list():
    country_file_name = "WDICountry_whitelist.csv"
    df = pd.read_csv(data_prefix + country_file_name, index_col=0)

    return df.index.to_numpy()


def load_WDI():
    '''
    loads WDI data csv file
    :return: pd.DataFrame
    '''
    WDI_data_file_name = "WDIData.csv"
    df = pd.read_csv(data_prefix + WDI_data_file_name, dtype=create_year_dict(range(1960, 2020), np.float))
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    df = df[df['Country Code'].isin(load_country_list())]

    return df


def load_WDI_Partials():
    sub_folder = 'data/'
    directory = data_prefix + sub_folder
    dfs = []
    for filename in os.listdir(directory):

        if filename.endswith(".csv"):
            print(filename)
            # print(data_prefix + sub_folder + filename)

            df = pd.read_csv(data_prefix + sub_folder + filename, header=2, skip_blank_lines=True, )
            # df.columns = name_cols
            dfs.append(df)

    return pd.concat(dfs)


def select_indicators(df, indicator_list):
    return df[df['Indicator Code'].isin(indicator_list)]


def get_name_replacements(columns=None):
    columns_replacements = columns.copy()
    d_types = {}
    if columns is None:
        columns = []
        columns_replacements = []
    for i in range(1980, 2021):
        columns.append("{} [YR{}]".format(i, i))
        columns_replacements.append(str(i))
        d_types["{} [YR{}]".format(i, i)] = 'float64'
    return columns, columns_replacements, d_types


def get_year_columns():
    columns = []
    for i in range(1980, 2021):
        columns.append(str(i))

    return columns


def turn_wide_into_long(wdi_data, id_vars):
    data = wdi_data.copy()
    data = pd.melt(data, id_vars=id_vars,
                   value_vars=get_year_columns(), var_name="Year",
                   value_name="Value")
    data[["Value", "Year"]] = data[["Value", "Year"]].apply(pd.to_numeric)

    return data


def missing_values_per_country_total(wdi_data):
    d = wdi_data.isnull().sum(axis=1)
    return d.sum(level='Country Code', axis=0)


def missing_values_per_country_indicator(wdi_data):
    for i in wdi_data.columns:
        d = wdi_data[i].isnull().sum(level='Country Code', axis=0)
        # print (i)
        # print (d)
        return d


def handle_missing_values(incomplete_data):
    complete_data = incomplete_data.dropna(
        subset=['Scientific and technical journal articles', 'Patent applications total, per capita'], how='all')
    print(complete_data.isnull().sum())  # Sum of nans for each indicator
    return complete_data


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    print(names)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def reframe(data, lag=3):
    values = data.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, lag, 1)
    return reframed, scaler


def do_lstms(wdi_data, country_code):
    train = wdi_data.copy().loc[country_code]['2000':'2020'].reset_index()
    print(train.columns)
    train.timestamp = pd.to_datetime(train.Year, format='%Y')
    train.index = train.timestamp
    train.drop('Year', axis=1, inplace=True)
    train.fillna(0, inplace=True)
    n_features = 27
    n_years = 3

    reframed, scaler = reframe(train, n_years)

    # split into train and test sets
    values = reframed.values
    n_train_years = 10
    train = values[:n_train_years, :]
    test = values[n_train_years:, :]
    # split into input and outputs
    n_obs = n_years * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_years, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_years, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    print(history.history)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # make a prediction

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_years * n_features))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -26:]), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -26:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    return model


def main():
    wdi_data = load_WDI_Partials()

    train = turn_wide_into_long(wdi_data, ['Country Code', 'Indicator Name'])

    wdi_data = wdi_data.set_index(['Country Code', 'Indicator Name'])
    wdi_data.drop(['Country Name', 'Indicator Code'], axis=1,
                  inplace=True)  # Drop Country Name as it is not required because of the country code
    wdi_data = wdi_data.sort_index(level=0)  # Sort dataframe by Country Code

    wdi_data = wdi_data.stack().unstack(
        level=1)  # Restack the dataframe so that the rows are the years and the columns are the indicators

    # Divide output parameters by population of respective year for better comparability
    wdi_data['Scientific and technical journal articles'] = wdi_data['Scientific and technical journal articles'].div(
        wdi_data['Population, total'], axis=0)
    wdi_data['Patent applications, residents'] = wdi_data['Patent applications, residents'].div(
        wdi_data['Population, total'], axis=0)
    wdi_data['Patent applications, nonresidents'] = wdi_data['Patent applications, nonresidents'].div(
        wdi_data['Population, total'], axis=0)

    # Sum patent applications of residents and non residents and drop the two columns
    wdi_data['Patent applications total, per capita'] = wdi_data['Patent applications, residents'] + wdi_data[
        'Patent applications, nonresidents']
    wdi_data.drop(['Patent applications, residents', 'Patent applications, nonresidents'], axis=1, inplace=True)

    # Move output parameters to the right side of the dataframe for better visibility
    wdi_data = wdi_data[[c for c in wdi_data if c not in ['Patent applications total, per capita',
                                                          'Scientific and technical journal articles']]
                        + ['Patent applications total, per capita', 'Scientific and technical journal articles']]

    # train = wdi_data.reset_index()

    wdi_data.index.names = ['Country Code', 'Year']
    ## missing data and stuff
    # d = (missing_values_per_country_total(wdi_data))
    # d.plot.hist(bins=20)

    # do_time_series_forecasting(wdi_data, 'AUT')
    model = do_lstms(wdi_data.drop(['Patent applications total, per capita'], axis=1), 'AUT')

    validation_data = wdi_data.copy().loc['DEU']['2000':'2020'].reset_index()
    print(validation_data.columns)
    validation_data.drop(['Patent applications total, per capita'], axis=1)
    validation_data.reset_index(inplace=True)
    validation_data.timestamp = pd.to_datetime(validation_data.Year, format='%Y')
    validation_data.index = validation_data.timestamp
    validation_data.drop('Year', axis=1, inplace=True)
    validation_data.fillna(0, inplace=True)
    print(validation_data.head())

    reframed, scaler = reframe(validation_data, 3)
    test = reframed.values
    print(test.shape)

    n_features = 27
    n_years = 3
    n_obs = n_years * n_features
    test_X, test_y = test[:, :n_obs], test[:, -26]

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], 3 * 27))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -26:]), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -26:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    plt.figure()
    plt.plot(inv_y)
    plt.plot(test_y)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #    print(d[d < 1000].sort_values())


def adf_test(testdata):
    dftest = adfuller(testdata, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def do_stabilization_test(series, title):
    # plt.figure()
    # orig = plt.plot(series, color='blue', label='Original')
    # mean = plt.plot(series.rolling(5).mean(), color='red', label='Rolling Mean')
    # std = plt.plot(series.rolling(5).std(), color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title(title + ' - Rolling Mean & Standard Deviation')
    # plt.show(block=False)
    adf_test(series.dropna())


def do_acf_pacf_plots(journals_avg_diff, title):
    journals_avg_diff.dropna(inplace=True)
    plt.figure()
    lag_acf = acf(journals_avg_diff, nlags=5)
    lag_pacf = pacf(journals_avg_diff, nlags=5, method='ols')
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(journals_avg_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(journals_avg_diff)), linestyle='--', color='gray')
    plt.title(title + '- Autocorrelation Function')
    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(journals_avg_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(journals_avg_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()


def fit_ARIMA(series):
    plt.figure()
    model = ARIMA(series, order=(2, 1, 0))
    results = model.fit()
    plt.plot(series)
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f' % sum((results.fittedvalues - series) ** 2))

    # predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
    # predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    # predictions_ARIMA_log = pd.Series(series.iloc[0], index=series.index)
    # predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

    return results.fittedvalues


def do_time_series_forecasting(wdi_data, country_code):
    train = wdi_data.copy().loc[country_code].reset_index()
    train.timestamp = pd.to_datetime(train.Year, format='%Y')
    train.index = train.timestamp
    train.drop('Year', axis=1, inplace=True)

    journals = train['Scientific and technical journal articles'].ffill()
    print(journals['2000':'2020'].head(25))
    journals.dropna(inplace=True)
    do_stabilization_test(journals, 'journals')

    patents = train['Patent applications total, per capita'].ffill()
    patents.dropna(inplace=True)
    print('patents shift')
    ts_log = patents
    patents_avg_diff = ts_log - ts_log.shift(1)
    do_stabilization_test(patents_avg_diff, 'patents shift diff')

    do_stabilization_test(patents_avg_diff, 'patents')
    # do_acf_pacf_plots(patents_avg_diff, 'patents')
    fitted = fit_ARIMA(patents_avg_diff)
    # fitted = fitted + fitted.rolled(5).mean()
    plt.figure()
    plt.plot(patents)
    plt.plot(fitted)
    plt.title('RMSE: %.4f' % np.sqrt(sum((fitted - patents) ** 2) / len(patents)))

    # ##### stabilize
    # print('shift')
    # ts_log = journals
    # journals_avg_diff = ts_log - ts_log.shift(1)
    # do_stabilization_test(journals_avg_diff, 'journals shift diff')
    # ##### stabilize
    # print('log shift')
    # ts_log = np.log(journals)
    # journals_avg_diff = ts_log - ts_log.shift(1)
    # do_stabilization_test(journals_avg_diff, 'journals shift diff')
    ##### stabilize
    print('rolling')
    ts_log = journals
    journals_avg_diff = ts_log - ts_log.rolling(5).mean()
    do_stabilization_test(journals_avg_diff, 'journals avg diff')
    # -> let's roll with this:::
    # do_acf_pacf_plots(journals_avg_diff, 'journals')
    fitted = fit_ARIMA(journals_avg_diff)
    fitted = fitted + fitted.rolling(5).mean()
    plt.figure()
    plt.plot(journals)
    plt.plot(fitted)
    plt.title('RMSE: %.4f' % np.sqrt(sum((fitted - journals) ** 2) / len(journals)))

    # ##### stabilize
    # print('rolling log')
    # ts_log = np.log(journals)
    # journals_avg_diff = ts_log - ts_log.rolling(5).mean()
    # do_stabilization_test(journals_avg_diff, 'journals log avg diff')
    #
    # ##### stabilize
    # print('ewm ')
    # ts_log = journals
    # journals_avg_diff = ts_log - ts_log.ewm(0.5).mean()
    # do_stabilization_test(journals_avg_diff, 'journals ewm diff')
    # ##### stabilize
    # print('ewm log')
    # ts_log = np.log(journals)
    # journals_avg_diff = ts_log - ts_log.ewm(0.5).mean()
    # do_stabilization_test(journals_avg_diff, 'journals log ewm diff')


if __name__ == '__main__':
    main()
    plt.show()
