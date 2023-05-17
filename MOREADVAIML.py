import sys
import numpy as np
import pandas as pd
from binance.client import Client
import logging
import time
from binance.exceptions import BinanceAPIException, BinanceRequestException
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from keras.layers import Dropout, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

tf.config.run_functions_eagerly(True)


api_key = 'PW1Cl5lFEP5JsGDoljgrvrddGA5PqPB4T6d0rTt8so0GR0RqpEhX4mfgoOhDtJMF'
api_secret = 'secret-key'

client = Client(api_key, api_secret, tld='us')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def fetch_recent_klines(symbol, interval):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval)
        return klines
    except (BinanceAPIException, BinanceRequestException) as e:
        logging.error(f"Error occurred while fetching recent klines for {symbol}: {e}")


def prepare_data(symbol, interval='30m'):
    klines = fetch_recent_klines(symbol, interval)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df.drop(['open_time', 'open', 'high', 'low', 'close_time', 'quote_asset_volume', 'number_of_trades',
             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['close']])

    return scaled_data, scaler, close_scaler





def create_lstm_model(input_shape):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=10, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Second LSTM layer
    model.add(LSTM(units=10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Third LSTM layer
    model.add(LSTM(units=10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Fourth LSTM layer
    model.add(LSTM(units=10))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Dense output layer with linear activation function
    model.add(Dense(1, activation='linear'))

    # Compile the model with the Adam optimizer and mean squared error loss
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model




def train_model(model, data, epochs=100, batch_size=32):
    x_train, y_train = [], []
    for i in range(60, len(data)):
        x_train.append(data[i - 60:i, 0])
        y_train.append(data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Split the data into training and validation sets
    x_train, x_val = x_train[:-60], x_train[-60:]
    y_train, y_val = y_train[:-60], y_train[-60:]

    # Set up early stopping and model checkpointing
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])



def predict_future_prices(model, data, close_scaler):
    inputs = data[len(data) - 60:, 0]
    inputs = inputs.reshape(-1, 1)
    inputs = close_scaler.transform(inputs)

  #  print("Inputs shape after transformation:", inputs.shape)
 #   print("Inputs content after transformation:", inputs)

    X_test = np.array([inputs[:, 0]])  # Convert the inputs to a 3D array with shape (1, 60, 1)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  #  print("X_test shape after reshaping:", X_test.shape)
   # print("X_test content after reshaping:", X_test)

    predicted_price = model.predict(X_test)
    predicted_price = close_scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]


def get_symbol_info(symbol):
    try:
        info = client.get_symbol_info(symbol)
        return info
    except (BinanceAPIException, BinanceRequestException) as e:
        logging.error(f"Error occurred while fetching info for {symbol}: {e}")



def execute_trades(pair, current_price, predicted_price, balance):
    risk_percentage = 0.5
    pair_info = get_symbol_info(pair)
    min_notional = float([x['minNotional'] for x in pair_info['filters'] if x['filterType'] == 'MIN_NOTIONAL'][0])
    step_size = float([x['stepSize'] for x in pair_info['filters'] if x['filterType'] == 'LOT_SIZE'][0])
    min_qty = .001

    usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])

    if predicted_price > current_price:
        try:
            risk_amount = usdt_balance * risk_percentage
            quantity = risk_amount / current_price
            quantity = np.round(quantity - (quantity % step_size), decimals=int(np.log10(1/step_size)))
            if quantity < min_qty:
                logger.info(f"Calculated quantity {quantity} is less than the minimum allowed quantity {min_qty}")
                return

            order = client.order_market_buy(
                symbol=pair,
                quantity=(quantity)
            )
            logging.info(f"Buying {pair}")
            logging.info(f"Buy order details: {order}")
        except (BinanceAPIException, BinanceRequestException) as e:
            print(quantity)
            logging.error(f"Error occurred while placing buy order for {pair}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while placing buy order for {pair}: {e}")
    else:
        try:
            asset_balance = client.get_asset_balance(asset=pair[:3])
            quantity = float(asset_balance['free'])
            quantity = np.round(quantity - (quantity % step_size), decimals=int(np.log10(1/step_size)))

            if quantity < min_qty:
                logger.info(f"Calculated quantity {quantity} is less than the minimum allowed quantity {min_qty}")
                return

            order = client.order_market_sell(
                symbol=pair,
                quantity=quantity
            )
            logging.info(f"Selling {pair}")
            logging.info(f"Sell order details: {order}")

        except (BinanceAPIException, BinanceRequestException) as e:
            logging.error(f"Error occurred while placing sell order for {pair}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while placing sell order for {pair}: {e}")



def main():
    pairs = ['BTCUSDT', 'BNBUSDT'] 
    models = {}
    iteration_count = 0
    RECALIBRATE_AFTER_ITERATIONS = 10  

    while True:
        predicted_prices = {}
        current_prices = {}

        for pair in pairs:
            try:
                data, scaler, close_scaler = prepare_data(pair)
                # Recalibrate model after every N iterations
                if iteration_count % RECALIBRATE_AFTER_ITERATIONS == 0 or pair not in models:
                    model = create_lstm_model((60, 1))
                    train_model(model, data)
                    models[pair] = model
                else:
                    model = models[pair]
                future_price = predict_future_prices(model, data, close_scaler)
                predicted_prices[pair] = future_price
                ticker = client.get_symbol_ticker(symbol=pair)
                current_price = float(ticker['price'])
                current_prices[pair] = current_price
            except Exception as e:
                logging.error(f"An error occurred while processing pair {pair}: {e}")

        max_profit_pair = max(predicted_prices, key=lambda x: (
                                                                      predicted_prices[x] - current_prices[x]) /
                                                              current_prices[x])
        balances = get_balances_for_pairs(pairs)

        if not balances:
            logging.error("No balances available for trading pairs. Stopping the bot.")
            break

        max_balance_pair = max(balances, key=balances.get)

        if max_profit_pair == max_balance_pair:
            logging.info(f"Holding {max_balance_pair} as it has the highest predicted profit")
        else:
            balance = float(client.get_asset_balance(asset=max_balance_pair[:3])['free'])
            logging.info(f"Current price of {max_profit_pair}: {current_prices[max_profit_pair]:.8f}")
            other_pair = [p for p in pairs if p != max_profit_pair][0]
            other_balance = float(client.get_asset_balance(asset=other_pair[:3])['free'])
            execute_trades(max_profit_pair, current_prices[max_profit_pair], predicted_prices[max_profit_pair], other_balance)
            logging.info(f"Predicted future price of {max_profit_pair}: {predicted_prices[max_profit_pair]:.8f}")

        time.sleep(300)  # Delay between iterations
        iteration_count += 1  # Increment the iteration count

if __name__ == '__main__':
    main()

    main()

