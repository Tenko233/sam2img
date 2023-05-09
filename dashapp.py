import base64
import datetime

from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback
import PIL

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='SAM to PSD')

image_display = html.Div([
    dcc.Upload([
        'Drag and Drop or Select a Picture'
    ],
        id='upload-image'),

    html.Div(id='output-image-upload')
])


def parse_contents(contents, filename, date):
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@callback(
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('upload-image', 'last_modified')
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children
