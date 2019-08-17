# -*- coding: utf-8 -*-
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import cv2 as cv 
import numpy as np
import plotly.graph_objects as go
from heatmap import *

external_stylesheets = [dbc.themes.BOOTSTRAP, './assetst/style.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#Load image
image_filename = 'field.png' # replace with your own image

#Process image and apply color masks
img = cv.imread('field.png') 
# img = generate_heatmap((512, 512), 0, 100)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
dst = cv.blur(img, (5,5))
healthy_range = (np.asarray([ 42, 45, 13]), np.asarray([71,84,51]))   # white!   # yellow! note the order
warning_range = (np.asarray([99, 99, 85]), np.asarray([146, 166, 152]))   
healthy_mask = cv.inRange(dst, healthy_range[0], healthy_range[1])
warning_mask = cv.inRange(dst, warning_range[0], warning_range[1])
result = cv.bitwise_and(img, img, mask=healthy_mask)
img[np.where(healthy_mask)] = [17,122,11]
img[np.where(warning_mask)] = [0,0,255]

healthy_perc = np.sum(healthy_mask/255)/(img.shape[0]*img.shape[1])
warning_perc = np.sum(warning_mask/255)/(img.shape[0]*img.shape[1])

cv.imwrite('masked_image.jpg', img)

encoded_image = base64.b64encode(open('masked_image.jpg', 'rb').read())

labels = ['Sano', 'En peligro']
values = [healthy_perc/(warning_perc + healthy_perc), warning_perc/(warning_perc + healthy_perc)]

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])


navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                dbc.Col(dbc.NavbarBrand("DashBoard", className="ml-2")),
                align="center",
                no_gutters=True,
            )
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    id='top'
)

img = html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())))

tab1_content = dbc.Card(
    dbc.CardImg(src='data:image/png;base64,{}'.format(encoded_image.decode())),
    className="mt-3 p-3 display-card"
)



content = dbc.Row(
    [
        dbc.Col([
            dbc.Card([
                html.H3('Visual'),
                html.Hr(),
                dbc.Tabs(
                [
                    dbc.Tab(tab1_content, label="Mapa"),
                    # dbc.Tab(tab2_content, label="Tab 2"),
                ], className='p-0 m-0', id='mapa'
            )
                ], className='full-card'
            ),
        ], width=4, className='content-section'),
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(
                        id='pie-charts',
                        figure=fig,
                        className='pie'
                    ), className='graph-card')
                ),
                dbc.Col(
                    dbc.Card(
                        dcc.Graph(
                        id='pie-chart',
                        figure=fig,
                        className='pie'
                    ), className='graph-card')
                )
            ], className='p-2'),
            dbc.Row([
                dbc.Col(
                    dbc.Card(dbc.CardBody('xd'))
                )
            ], className='p-2')
        ], width=6, className='content-section'),
        dbc.Col([
            dbc.Card([
                html.H3('Filtros'),
                html.Hr(),
                dbc.Form([
                    dbc.FormGroup([
                        dbc.Checkbox(id='check1', checked=False),
                        dbc.Label('  Humedad', html_for='check1'),
                        html.Div([
                            html.P('Rango: '),
                            dbc.Input(className='input-par', id='range1a', type='number', max=100, min=0, value=0),
                            html.Span(' , '),
                            dbc.Input(className='input-par', id='range1b', type='number', max=100, min=0, value=100)
                        ],id='check1div', className='hidden'),
                        html.Br(),
                        dbc.Checkbox(id='check2', checked=False),
                        dbc.Label('  pH', html_for='check2'),
                        html.Div([
                            html.P('Rango: '),
                            dbc.Input(className='input-par', id='range2a', type='number', max=100, min=0, value=0),
                            html.Span(' , '),
                            dbc.Input(className='input-par', id='range2b', type='number', max=100, min=0, value=100)
                        ], id='check2div', className='hidden'),
                        html.Br(),
                        dbc.Checkbox(id='check3', checked=False),
                        dbc.Label('  Color', html_for='check3'),
                        html.Div([
                            html.P('Rango: '),
                            dbc.Input(className='input-par', id='range3a', type='number', max=100, min=0, value=0),
                            html.Span(' , '),
                            dbc.Input(className='input-par', id='range3b', type='number', max=100, min=0, value=100)
                        ], id='check3div', className='hidden'),
                    ])
                ], id='param-container'),
                        dbc.Button('Aplicar', className='cust-btn px-auto', id='apply-btn')
                ], className='full-card'
            ),
        ], width=2, className='content-section'),
        
    ], 
    className='main-container'
)

app.layout = html.Div(children=[
    navbar,
    content
    # dcc.Graph(
    #     id='pie-chart',
    #     figure=fig
    # )
])

@app.callback(
    Output('mapa', 'children'),
    [Input('apply-btn', 'n_clicks'),
    Input('check1', 'checked'),
    Input('check2', 'checked'),
    Input('check3', 'checked'),
    Input('range1a', 'value'),
    Input('range2a', 'value'),
    Input('range3a', 'value'),
    Input('range1b', 'value'),
    Input('range2b', 'value'),
    Input('range3b', 'value'),
    ])
def update_output(n_clicks, use_hum, use_ph, use_cal, min_hum, max_hum, min_ph, max_ph, min_cal, max_cal):
    return dbc.Tab(tab1_content, label="Mapa")

@app.callback(
    Output(component_id='check1div', component_property='className'),
    [Input(component_id='check1', component_property='checked')]
)
def on_check1(checked):
    if not checked:
        return 'hidden'
@app.callback(
    Output(component_id='check2div', component_property='className'),
    [Input(component_id='check2', component_property='checked')]
)
def on_check2(checked):
    if not checked:
        return 'hidden'
@app.callback(
    Output(component_id='check3div', component_property='className'),
    [Input(component_id='check3', component_property='checked')]
)
def on_check3(checked):
    if not checked:
        return 'hidden'


if __name__ == '__main__':
    app.run_server(debug=True)