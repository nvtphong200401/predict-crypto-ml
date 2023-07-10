import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
import requests
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def extract_window_data(df, window_len=10, zero_base=True):
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



valid = df

model=load_model("crypto_model.h5")

X_test= extract_window_data(df)

closing_price=model.predict(X_test)
# valid['Predictions']=closing_price



app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
        # Tab 1
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data',children=[
            html.Div([
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid[target_col],
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
                        )
                    }

                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=closing_price,
                                mode='markers'
                            )

                        ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate'}
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


# @app.callback(Output('highlow', 'figure'),
#               [Input('my-dropdown', 'value')])
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["High"],
#                      mode='lines', opacity=0.7, 
#                      name=f'High {dropdown[stock]}',textposition='bottom center'))
#         trace2.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Low"],
#                      mode='lines', opacity=0.6,
#                      name=f'Low {dropdown[stock]}',textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Price (USD)"})}
#     return figure


# @app.callback(Output('volume', 'figure'),
#               [Input('my-dropdown2', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Volume"],
#                      mode='lines', opacity=0.7,
#                      name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data, 
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M',
#                                                        'step': 'month', 
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Transactions Volume"})}
#     return figure


if __name__=='__main__':
    app.run_server(debug=True)