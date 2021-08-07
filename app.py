#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import networkx as nx
import plotly.graph_objs as go

import pandas as pd
from colour import Color
from datetime import datetime
from textwrap import dedent as d
import json
import subprocess
import numpy as np
import sys

# import the css template, and pass the css template into dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Visualize Persian Knowledge Graph"

fname = "input"


def network_graph(cmd="venv/bin/python3.8 match.py"):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    print(f"done executing: \n{cmd}, p.communicate: {p.communicate()}")

    G = nx.DiGraph()
    with open('output.json', 'r') as f:
        for l in f.readlines():
            l = l.replace("\'", "\"")
            current_line = json.loads(l)

            for tup in current_line['tri']:
                # G.add_edge(tup['h'], tup['t'], label=str(tup['r']) + "_" + str(np.round(tup['c'], 2)))
                G.add_edge(tup['h'], tup['t'], label=str(tup['r'][0]))

    pos = nx.shell_layout(G)

    edge_x = []
    edge_y = []

    etext = [f'{w}' for w in list(nx.get_edge_attributes(G, 'label').values())]
    xtext = []
    ytext = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        xtext.append((x0 + x1) / 2)  # for edge text
        ytext.append((y0 + y1) / 2)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        textfont=dict(
            size=16),
        mode='lines')

    eweights_trace = go.Scatter(x=xtext, y=ytext, mode='text',
                                marker_size=0.5,
                                text=etext,
                                textfont=dict(
                                    size=16),
                                textposition='top center',
                                hovertemplate='weight: %{text}<extra></extra>')

    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textfont=dict(
            size=16),
        textposition="bottom center",
        text=list(G.nodes()),
        marker={'size': 50, 'color': 'Purple'}
    )

    figure = go.Figure(data=[edge_trace, node_trace, eweights_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin={'b': 40, 'l': 40, 'r': 40, 't': 40},
                           xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                           yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
                           height=700,
                           annotations=[
                               dict(
                                   ax=(pos[edge[0]][0] + pos[edge[1]][0]) / 2,
                                   ay=(pos[edge[0]][1] + pos[edge[1]][1]) / 2, axref='x', ayref='y',
                                   x=(pos[edge[1]][0] * 9 + pos[edge[0]][0]) / 10,
                                   y=(pos[edge[1]][1] * 9 + pos[edge[0]][1]) / 10, xref='x', yref='y',
                                   showarrow=True,
                                   arrowhead=3,
                                   arrowsize=4,
                                   arrowwidth=1,
                                   opacity=1
                               ) for edge in G.edges]
                       ))
    return figure


######################################################################################################################################################################
# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([

    html.H1("Visualize Persian Knowledge Graph", style={'text-align': 'center'}),
    html.Div(dcc.Graph(id='my_graph', figure={})),
    html.Br(),
    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0)

])


# Connect the Plotly graphs with Dash Components
@app.callback(
    dash.dependencies.Output(component_id='my_graph', component_property='figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
)
def update_graph(n_clicks):
    return network_graph()


if __name__ == '__main__':
    app.run_server(debug=True)
