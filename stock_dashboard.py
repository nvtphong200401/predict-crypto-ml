import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

from keras.models import load_model
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np

window_len = 10

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def extract_window_data(df, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=200')

df = pd.DataFrame(json.loads(res.content)['Data'])
df = df.set_index('time')
df.index = pd.to_datetime(df.index, unit='s')
target_col = 'close'
df.drop(["conversionType", "conversionSymbol", "high", "low", "volumeto"], axis = 'columns', inplace = True)



targets = df[target_col][window_len:]

model=load_model("crypto_model.h5")

X_test = extract_window_data(df)

preds = model.predict(X_test).squeeze()
# valid['Predictions']=closing_price
preds = df[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)


app.layout = html.Div([
   
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
        # Tab 1
        dcc.Tab(children=[
            html.Div([
                html.H2("LSTM predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    figure={
                        "data":[
                            go.Scatter(
                                x=targets.index,
                                y=targets,
                                mode='lines',
                                text='Actual  closing price',
                                name='Actual'
                            ),
                            go.Scatter(
                                x=preds.index,
                                y=preds,
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(167, 167, 167, 0.12)',
                                text='Predicted closing price',
                                name='Predicted'
                            )
                        ],
                        "layout":go.Layout(
                            title='BTC - USD',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Price (USD)'}
                        )
                    }
                )               
            ])                
        ]),
        # Tab 2
        # dcc.Tab(label='Facebook Stock Data', children=[
        #     html.Div([
        #         html.H1("Facebook Stocks High vs Lows", 
        #                 style={'textAlign': 'center'}),
              
        #         dcc.Dropdown(id='my-dropdown',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple','value': 'AAPL'}, 
        #                               {'label': 'Facebook', 'value': 'FB'}, 
        #                               {'label': 'Microsoft','value': 'MSFT'}], 
        #                      multi=True,value=['FB'],
        #                      style={"display": "block", "margin-left": "auto", 
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='highlow'),
        #         html.H1("Facebook Market Volume", style={'textAlign': 'center'}),
         
        #         dcc.Dropdown(id='my-dropdown2',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple','value': 'AAPL'}, 
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft','value': 'MSFT'}], 
        #                      multi=True,value=['FB'],
        #                      style={"display": "block", "margin-left": "auto", 
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='volume')
        #     ], className="container"),
        # ])


    ])
])

if __name__=='__main__':
    app.run_server(debug=True)