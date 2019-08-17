# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import cv2 as cv 
import numpy as np
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Load image
image_filename = 'field.png' # replace with your own image

#Process image and apply color masks
img = cv.imread('field.png') 
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

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
    ]),
    dcc.Graph(
        id='pie-chart',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)