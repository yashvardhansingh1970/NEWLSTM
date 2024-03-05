import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from keras.models import load_model  # Corrected import statement
import streamlit as st
import plotly.graph_objs as go

st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker','AAPL')

# Set start and end dates
start_date = '1986-03-19'
end_date = '2024-02-24'

if start_date <= end_date:
    # Fetch historical stock data using yfinance
    try:
        # Download the historical data
        df = yf.download(user_input, start=start_date, end=end_date)

        # Display the historical data
        st.write(f"Historical data for {user_input} from {start_date} to {end_date}:")
        st.write(df)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.error("End date must be greater than or equal to start date.")


st.subheader('Closing Price vs Time Chart')
trace = go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Closing Price')

layout = go.Layout(
    title=f"{user_input} Closing Price",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Price (USD)")
)

fig = go.Figure(data=[trace], layout=layout)

# Show the interactive plot in the Jupyter Notebook
st.plotly_chart(fig)

df.reset_index(inplace=True)

import datetime

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

datetime_object = str_to_datetime('1986-03-19')
# # datetime_object
#
df['Date'] = pd.to_datetime(df['Date'])
#
df.index = df.pop(df.columns[0])


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    print("ret_df is ", ret_df)

    return ret_df


# Start day second time around: '2022-08-10'
windowed_df = df_to_windowed_df(df,
                                '1986-03-19',
                                '2024-02-24',
                                n=3)
# windowed_df


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


dates, X, y = windowed_df_to_date_X_y(windowed_df)

dates.shape, X.shape, y.shape
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

model = load_model('LSTM_model.h5')

model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=100)


# plt.plot(dates_train, y_train)
# plt.plot(dates_val, y_val)
# plt.plot(dates_test, y_test)
#
# plt.legend(['Train', 'Validation', 'Test'])
#
#
# plt.plot(dates_train, y_train, label='Training Data')
#
# # Plot the validation data
# plt.plot(dates_val, y_val, label='Validation Data')
#
# # Plot the testing data
# plt.plot(dates_test, y_test, label='Testing Data')

# Set the labels and title
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.title('Plot of Training, Validation, and Testing Data')
# plt.legend()
#
# # Display the plot using Streamlit's st.pyplot()
# st.pyplot()



train_predictions = model.predict(X_train).flatten()


trace_train_pred = go.Scatter(x=dates_train, y=train_predictions, mode='lines', name='Training Predictions')
trace_train_obs = go.Scatter(x=dates_train, y=y_train, mode='lines', name='Training Observations')
trace_val_pred = go.Scatter(x=dates_val, y=val_predictions, mode='lines', name='Validation Predictions')
trace_val_obs = go.Scatter(x=dates_val, y=y_val, mode='lines', name='Validation Observations')
trace_test_pred = go.Scatter(x=dates_test, y=test_predictions, mode='lines', name='Testing Predictions')
trace_test_obs = go.Scatter(x=dates_test, y=y_test, mode='lines', name='Testing Observations')

# Combine the traces into a data list
data = [trace_train_pred, trace_train_obs, trace_val_pred, trace_val_obs, trace_test_pred, trace_test_obs]

# Layout for the plot
layout = go.Layout(title='Interactive Plot of Predictions and Observations',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Close Price'),
                   legend=dict(orientation='h'))

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Display the figure using Plotly in Streamlit
st.plotly_chart(fig)


future_dates = pd.date_range(start=dates_test[-1], periods=10, freq='D')

# Reshape the last observed sequence to match the input shape of the model
last_sequence = X_test[-1].reshape((1, X_test.shape[1], 1))

# Generate future predictions
future_predictions = []
for _ in range(10):
    # Predict the next value
    next_prediction = model.predict(last_sequence)

    # Append the prediction to the list of future predictions
    future_predictions.append(next_prediction[0])

    # Update last_sequence by removing the first element and appending the predicted value
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1][-1] = next_prediction

# Convert the predictions to numpy array
future_predictions = np.array(future_predictions).flatten()
trace_train = go.Scatter(x=dates_train, y=y_train, mode='lines', name='Training Observations')
trace_val = go.Scatter(x=dates_val, y=y_val, mode='lines', name='Validation Observations')
trace_test = go.Scatter(x=dates_test, y=y_test, mode='lines', name='Testing Observations')
trace_future = go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions', line=dict(dash='dash'))

# Combine the traces into a data list
data = [trace_train, trace_val, trace_test, trace_future]

# Layout for the plot
layout = go.Layout(title='Interactive Plot of Observations and Future Predictions',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Close Price'),
                   legend=dict(orientation='h'))

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Display the figure using Plotly in Streamlit
st.plotly_chart(fig)


