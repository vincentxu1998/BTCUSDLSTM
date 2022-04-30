import plotly.graph_objects as go # or plotly.express as px
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
from plotly.subplots import make_subplots

# fig.add_trace( ... )
# fig.update_layout( ... )

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
#
# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
#
# app.run_server(debug=True, use_reloader=False)

def plot_candle(df):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5,0.1,0.2,0.2],
                        )

    fig1 = go.Candlestick(x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price',
                    showlegend=False)

    colors = ['green' if row['open'] - row['close'] <= 0
              else 'red' for index, row in df.iterrows()]

    fig2 = go.Bar(x=df.index, y=df['volume'], showlegend=False, marker_color=colors)

    colors = ['green' if row['macdhist'] >= 0
              else 'red' for index, row in df.iterrows()]
    fig3 = go.Bar(x=df.index,
                  y=df['macdhist'],
                  marker_color=colors,
                  showlegend=False)
    fig4 = go.Scatter(x=df.index,
                      y=df['macd'],
                      line=dict(color='black', width=2),
                      showlegend=False)
    fig5 = go.Scatter(x=df.index,
                      y=df['macdsignal'],
                      line=dict(color='blue', width=1),
                      showlegend=False)


    fig.add_trace(fig1, row=1, col=1)
    fig.add_trace(fig2, row=2, col=1)
    fig.add_traces([fig3, fig4, fig5], rows=[3,3,3], cols=[1,1,1])

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", showgrid=False, row=3, col=1)
    fig.update_yaxes(title_text="Stoch", row=4, col=1)

    fig.update(layout_xaxis_rangeslider_visible=True)
    fig.update_yaxes(autorange=True)
    fig.show()