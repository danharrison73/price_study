import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from matplotlib.patches import Rectangle


def convert_date(date_string):
    return datetime.strptime(str(date_string), "%Y%m%d")


def clean_df():
    at = pd.read_csv('data/atandt.csv')

    at = at[at.columns[at.nunique() > 7]]

    # Convert date row to datetime object
    at['date'] = at['DATE'].apply(lambda x: convert_date(x))

    # Ensure data is sorted by date
    at.sort_values('date', inplace=True)

    # Calculate daily returns and append as a new column
    at['daily_return'] = (at['PRC'].pct_change())
    at = at.dropna()

    at['abs_return'] = at['daily_return'].abs()

    # CREATE NEW ROW FOR LIQUIDITY
    at['liq'] = at['ASKHI'] - at['BIDLO']

    # Calculate rolling volatility (e.g., over a 20-day window)
    at['volatility'] = at['daily_return'].rolling(window=20).std()

    # at['VOL'] = (at['VOL'] - at['VOL'].mean()) / (at['VOL'].max() - at['VOL'].min())

    # Null columns are dropped again (where there is a NaN value for volatility)
    at = at.dropna()

    at['next_day_abs_return'] = at['abs_return'].shift(-1)
    at = at.dropna()

    rolling_window_size = 20  # or whatever window size you prefer
    at['rolling_volume_mean'] = at['VOL'].rolling(window=rolling_window_size).mean()

    at['rolling_liq_mean'] = at['liq'].rolling(window=rolling_window_size).mean()

    # drop the first value (as there is no daily return)
    at = at.dropna()

    df = at[['date','daily_return', 'liq', 'VOL', 'volatility', 'next_day_abs_return', 'rolling_volume_mean', 'rolling_liq_mean']]

    # Set the style and color palette
    sns.set_theme(style="darkgrid")
    sns.set_palette("bright")

    at['abs_return'] = at['daily_return'].abs()

    sns.scatterplot(x='rolling_liq_mean', y='next_day_abs_return', data=df)
    plt.xlabel('Rolling Mean Liquidity')
    plt.ylabel('Next Day Absolute Return')
    plt.title('Relationship between Rolling Mean Liquidity and Next Day Absolute Return')
    plt.show()

    correlation_matrix = df.corr(numeric_only=True)
    ax = sns.heatmap(correlation_matrix, annot=True)
    # Highlight the next day magnitude of price change
    column_name = 'next_day_abs_return'
    column_index = correlation_matrix.columns.get_loc(column_name)

    # Adding a red rectangle around the column with the specified index
    rect = Rectangle((column_index, 0), 1, correlation_matrix.shape[0], linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    plt.show()

    return df


def visualise_data(df):
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(10,10))  # 1 row, 2 columns

    # Plot price over time
    axes[0].plot(df['date'], df['PRC'])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Price Over Time')
    axes[0].grid(True)

    # Plot volatility over time
    axes[1].plot(df['date'], df['volatility'])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Volatility')
    axes[1].set_title('Volatility Over Time')
    axes[1].grid(True)

    plt.tight_layout()  # Adjusts subplot params to fit the figure area
    plt.show()

def multiple_linear_regression(df):
    X = df[['VOL', 'liq', 'volatility', 'rolling_volume_mean', 'rolling_liq_mean']]
    y = df['next_day_abs_return']
    dates = df['date']

    X_train, X_test, y_train, y_test, test_dates = sequential_test_split(X, y, dates, test_size=0.3)

    # Create a pipeline that first applies polynomial transformation and then fits the linear regression
    model = LinearRegression()

    # Fit the model to the data (X_train and y_train are your training data)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    # print('Mean Squared Error:', mse)
    rmse = np.sqrt(mse)
    # print('Root Mean Squared Error:', rmse)

    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line
    plt.xlabel('Actual Volatility')
    plt.ylabel('Predicted Volatility')
    plt.title('Actual vs Predicted Volatility')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Actual Volatility', color='blue')
    plt.plot(test_dates, y_pred, label='Predicted Volatility', color='red')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('Actual vs Predicted Volatility Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {'Mean Squared Error':mse, 'Root Mean Squared Error':rmse}

def polynomial_regression(df, degree=2, test_size=0.3):
    X = df[['VOL', 'liq', 'volatility', 'rolling_volume_mean', 'rolling_liq_mean']]
    y = df['next_day_abs_return']
    dates = df['date']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test, y_train, y_test, test_dates = sequential_test_split(X, y, dates, test_size=test_size)

    # Create polynomial features and then a linear regression model
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()

    lasso_model = Lasso(alpha=0.00002)

    # Create a pipeline that first applies polynomial transformation and then fits the linear regression
    # model = make_pipeline(polynomial_features, linear_regression)
    model = make_pipeline(MinMaxScaler(), polynomial_features, lasso_model)

    # Fit the model to the data (X_train and y_train are your training data)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)
    rmse = np.sqrt(mse)
    print('Root Mean Squared Error:', rmse)

    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line
    plt.xlabel('Actual Volatility')
    plt.ylabel('Predicted Volatility')
    plt.title('Actual vs Predicted Volatility')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Actual Volatility', color='blue')
    plt.plot(test_dates, y_pred, label='Predicted Volatility', color='red')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('Actual vs Predicted Volatility Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def sequential_test_split(X, y, dates=None, test_size=0.2):
    # Calculate the index at which to split
    split_index = int((1 - test_size) * len(df))

    # Split the features (X)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    # Split the target variable (y)
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = dates.iloc[split_index:]

    if dates is not None:
        return X_train, X_test, y_train, y_test, test_dates
    else:
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = clean_df()

    # visualise_data(df)
    # multiple_linear_regression(df)

    polynomial_regression(df)