# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import cv2 as cv 
import numpy as np
import plotly.graph_objects as go
from heatmap import *    
# import chart_studio.plotly as py
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Load image
image_filename = 'field.png' # replace with your own image

#Process image and apply color masks
img = cv.imread('field2.png') 
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


heat, temp_data = generate_heatmap((img.shape[0], img.shape[1]), 18, 20)
humidity, hum_data = generate_heatmap((img.shape[0], img.shape[1]), 50, 80)

temp_range = [18.5, 19]
hum_range = [60, 75] 

heat = cv.cvtColor(heat, cv.COLOR_BGR2GRAY)
humidity = cv.cvtColor(humidity, cv.COLOR_BGR2GRAY)

#Join for temp range, hum range and warning pixels
heat[temp_range[1] > temp_data] = 255
heat[temp_range[0] < temp_data] = 255
heat[temp_range[0] > temp_data] = 0
heat[temp_range[1] < temp_data] = 0

humidity[hum_range[1] > hum_data] = 255
humidity[hum_range[0] < hum_data] = 255
humidity[hum_range[0] > hum_data] = 0
humidity[hum_range[1] < hum_data] = 0

print(img.shape[0]*img.shape[1])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(heat[i][j] == 255 and humidity[i][j] == 255): #
            img[i][j][0] = 255
            img[i][j][1] = 0
            img[i][j][2] = 0

        # else:
        #     img[i][j][0] = 0
        #     img[i][j][1] = 0
        #     img[i][j][2] = 0

# img[np.where(heat)] = [255, 0, 0]
# print(img)
# img = cv.imread('field2.png') 
# result = cv.addWeighted(heat, 0.4, img, 0.6,0.0)

temp_data = temp_data.flatten()
temp_data.sort()

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

chunks = chunkIt(temp_data, 20)
x = [chunk.mean() for chunk in chunks]

fig = go.Bar(x = x, y=chunks)

cv.imwrite('masked_image.jpg', img)
encoded_image = base64.b64encode(open('masked_image.jpg', 'rb').read())

labels = ['Sano', 'En peligro']
values = [healthy_perc/(warning_perc + healthy_perc), warning_perc/(warning_perc + healthy_perc)]

# fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
print(len(temp_data))
# plt.hist(temp_data)
# plt.title("Temperature Histogram")
# plt.xlabel("Temperature")
# plt.ylabel("Count")

# fig = plt.gcf()
# plotly_fig = tls.mpl_to_plotly( fig )

# fig = px.histogram(df, x="total_bill")

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
        figure = fig     
    ),
    # dcc.Graph(
    #     id='life-exp-vs-gdp',
    #     figure={
    #         'data': [
    #             go.Scatter(
    #                 x=[i for i in range(len(temp_data))],
    #                 y=temp_data,
    #                 # text=df[df['continent'] == i]['country'],
    #                 mode='markers',
    #                 opacity=0.7,
    #                 marker={
    #                     'size': 15,
    #                     'line': {'width': 0.5, 'color': 'white'}
    #                 },
    #                 # name=i
    #             ) #for i in df.continent.unique()
    #         ],
    #         'layout': go.Layout(
    #             xaxis={'type': 'log', 'title': 'GDP Per Capita'},
    #             yaxis={'title': 'Life Expectancy'},
    #             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
    #             legend={'x': 0, 'y': 1},
    #             # hovermode='closest'
    #         )
    #     }
    # )
])

if __name__ == '__main__':
    app.run_server(debug=True)