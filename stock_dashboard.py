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

endpoint = 'https://min-api.cryptocompare.com/data/histoday'

def get_chart_result(from_coin: str, to_coin: str, model_file: str):
    res = requests.get(endpoint + '?fsym=' + from_coin + '&tsym='+ to_coin +'&limit=200')

    df = pd.DataFrame(json.loads(res.content)['Data'])
    df = df.set_index('time')
    df.index = pd.to_datetime(df.index, unit='s')
    target_col = 'close'
    df.drop(["conversionType", "conversionSymbol", "high", "low", "volumeto"], axis = 'columns', inplace = True)



    targets = df[target_col][window_len-1:-1]



    model=load_model(model_file)

    X_test = extract_window_data(df)

    preds = model.predict(X_test).squeeze()

    preds = df[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    return targets, preds

def show_chart(title: str, actual, pred):
    return dcc.Graph(
                    figure={
                        "data":[
                            go.Scatter(
                                x=actual.index,
                                y=actual,
                                mode='lines',
                                text='Actual closing price',
                                name='Actual'
                            ),
                            go.Scatter(
                                x=pred.index,
                                y=pred,
                                mode='lines',
                                fill='tonexty',
                                fillcolor='rgba(167, 167, 167, 0.12)',
                                text='Predicted closing price',
                                name='Predicted'
                            )
                        ],
                        "layout":go.Layout(
                            title=title,
                            xaxis={'title':'Date'},
                            yaxis={'title':'Price (USD)'}
                        )
                    }
                )


actual_btc, pred_btc = get_chart_result(from_coin='BTC', to_coin='USD', model_file="crypto_btc_model.h5")
actual_eth, pred_eth = get_chart_result(from_coin='ETH', to_coin='USD', model_file="crypto_eth_model.h5")
actual_ada, pred_ada = get_chart_result(from_coin='ADA', to_coin='USD', model_file="crypto_ada_model.h5")




app.layout = html.Div([
   
    html.H1("Cryptocurrency Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
        # Tab 1
        dcc.Tab(children=[
            html.Div([
                html.H2("LSTM predicted closing price",style={"textAlign": "center"}),
                show_chart(title='BTC - USD', actual=actual_btc, pred=pred_btc),
                show_chart(title='ETH - USD', actual=actual_eth, pred=pred_eth),
                show_chart(title='ADA - USD', actual=actual_ada, pred=pred_ada)
                # dcc.Graph(
                #     figure={
                #         "data":[
                #             go.Scatter(
                #                 x=actual_btc.index,
                #                 y=actual_btc,
                #                 mode='lines',
                #                 text='Actual closing price',
                #                 name='Actual'
                #             ),
                #             go.Scatter(
                #                 x=pred_btc.index,
                #                 y=pred_btc,
                #                 mode='lines',
                #                 fill='tonexty',
                #                 fillcolor='rgba(167, 167, 167, 0.12)',
                #                 text='Predicted closing price',
                #                 name='Predicted'
                #             )
                #         ],
                #         "layout":go.Layout(
                #             title='BTC - USD',
                #             xaxis={'title':'Date'},
                #             yaxis={'title':'Price (USD)'}
                #         )
                #     }
                # )               
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