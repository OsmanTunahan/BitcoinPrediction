import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class Prediction:
    def __init__(self, prices):
        self.prices = prices

    def train(self):
        df = pd.DataFrame({'bitcoin_prices': self.prices})

        plt.plot(df.index, df['bitcoin_prices'], label='Bitcoin Prices')
        plt.xlabel('Days')
        plt.ylabel('Bitcoin Prices')
        plt.legend()
        plt.show()

        X = df.index.values.reshape(-1, 1)
        y = df['bitcoin_prices']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return [X_test, y_test, model]

    def prediction(self):
        X_test, y_test, model = self.train()
        y_pred = model.predict(X_test)

        plt.plot(X_test, y_test, label='True Prices')
        plt.plot(X_test, y_pred, label='Predicted Prices')
        plt.xlabel('Days')
        plt.ylabel('Bitcoin Prices')
        plt.legend()
        plt.show()

        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

if __name__ == '__main__':
    req = requests.get('https://api.coin-stats.com/v2/coin_chart/bitcoin?type=1m')
    res = req.json()

    if req.status_code != 200 or res['status'] != 'success':
        print('No data.')
        exit()

    newPrices = []
    for day, value in enumerate(res['data']):
        newPrices.append(value[1])

    Bitcoin = Prediction(newPrices)
    Bitcoin.prediction()
