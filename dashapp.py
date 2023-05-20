import base64
import datetime
import io
import json
import random

from PIL import Image
import plotly.express as px
import dash
from dash import dcc, html
from dash import Input, Output, State, Patch, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import dash_resusable_components as drc


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded))

    return image, filename


app = dash.Dash(__name__,
                title="SAM to IMG",
                external_stylesheets=[dbc.themes.QUARTZ],
                suppress_callback_exceptions=False)

upload_bar = html.Div([
    dcc.Upload(
        id='upload-image',
        multiple=False,
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'margin': '0px',
            'textAlign': 'center'
        }
    )
])

image_card = html.Div([
    html.Br(),
    html.Div([
        html.H5('uploaded image will be showed below')
    ], id='image-title'),

    dcc.Loading([
        dcc.Graph(
            id='image',
            figure={
                'layout': {
                    'height': 700,
                    'width': 700,
                    'margin': {'l': 0, 'b': 0, 't': 0, 'r': 0},
                    'xaxis': {'showgrid': False},
                    'yaxis': {'showgrid': False}
                }
            },
            config={
                'displaylogo': False,
                'scrollZoom': True,
                'modeBarButtonsToRemove': ['zoom2d'],
                'modeBarButtonsToAdd': []
            }
        ),

        dcc.Store(id='image-info', storage_type='memory')
    ], style={'display': 'flex', 'justify-content': 'center'})

])

model_selection = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Label('Model'),
            dbc.RadioItems(
                id='model-selection',
                options=['vit_h', 'vit_b', 'vit_l'],
                value='vit_h'
            )
        ]),

        dbc.Col([
            dbc.Label('Device'),
            dbc.RadioItems(
                id='device-selection',
                options=['cpu', 'cuda'],
                value='cpu'
            )
        ])
    ])
])

point_input_area = dbc.Collapse([
    dbc.Button('Add Point', id='add-point', color='secondary', className='mt-3'),
    html.Div(id='point-input')
], id='point-input-area', is_open=False)

box_input_area = dbc.Collapse([
    drc.GroupInput(
        index=0,
        labeled=False,
        placeholder_x='left edge x coordinate',
        placeholder_y='top edge y coordinate'
    ),
    html.Br(),

    drc.GroupInput(
        index=1,
        labeled=False,
        placeholder_x='right edge x coordinate',
        placeholder_y='bottom edge y coordinate'
    ),
    html.Br()
], id='box-input-area', is_open=False)

option_buttons = dbc.Row([
        dbc.Col([
            dbc.Button('Generate', id='generate', color='primary', className='mt-3',
                       style={'width': '100%'})
        ], width=8),
        dbc.Col([
            dbc.Button('View Prompts', id='view-prompts', color='info', className='mt-3',
                       style={'width': '100%'})
        ], width=4),
    ])

sam_options = dbc.Col([
    html.Div([
        dbc.Tabs([
            dbc.Tab(
                label='Point Prompt',
                tab_id='point'
            ),

            dbc.Tab(
                label='Box Prompt',
                tab_id='box'
            ),

            dbc.Tab(
                label='Auto Segmentation',
                tab_id='auto'
            ),

        ], id='mode', active_tab='point'),
    ], style={'display': 'flex', 'justify-content': 'center'}),
    html.Br(),

    model_selection,
    point_input_area,
    box_input_area,
    option_buttons,
    html.Br()

], style={'margin-right': '10px'})

layout = dbc.Container([
    html.Br(),
    html.H1('SAM to IMG', style={'text-align': 'center'}),
    html.Br(),

    dbc.Row([
        dbc.Col([
            upload_bar
        ], width=12, class_name='ml-0')
    ]),

    dbc.Row([
        dbc.Col([
            image_card,
        ], lg=6, md=12, ),

        dbc.Col([
            sam_options
        ], lg=6, md=12, class_name='mt-3')
    ]),

], fluid=True)


@app.callback(
    Output('image', 'figure'),
    Output('image-info', 'data'),
    Output('image-title', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('upload-image', 'last_modified'))
def upload_image(content, name, date):
    if content is None:
        raise PreventUpdate
    image, filename = parse_contents(content, name)
    size = image.size
    data = {'image': image, 'size': size}
    fig = px.imshow(image)
    fig.update_layout(
        width=800, height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode=False
    )
    info = [
        html.H5(f'uploaded {size[0]} × {size[1]} image'),
        html.H6(f'filename: {filename}'),
        html.H6(f'last modified: {datetime.datetime.fromtimestamp(date)}')
    ]

    return fig, data, info


@app.callback(
    Output('point-input-area', 'is_open'),
    Output('box-input-area', 'is_open'),
    Input('mode', 'active_tab')
)
def update_input_form(mode):
    match mode:
        case 'point':
            return True, False
        case 'box':
            return False, True
        case 'auto':
            return False, False


@app.callback(
    Output('point-input', 'children'),
    Input('add-point', 'n_clicks'),
    State('image-info', 'data')
)
def add_point_input(n_clicks, data):
    try:
        size = data['size']
    except TypeError:
        size = None
    print(size)
    input_boxes = Patch()
    if n_clicks is None:
        raise PreventUpdate
    else:
        new_input = drc.GroupInput(
            index=n_clicks,
            size=size,
            labeled=True,
            placeholder_x='x coordinate',
            placeholder_y='y coordinate'
        )
        input_boxes.append(new_input)
        input_boxes.append(html.Br())
    return input_boxes


@app.callback(
    Output({'type': 'box-x', 'index': 0}, 'max'),
    Output({'type': 'box-y', 'index': 0}, 'max'),
    Output({'type': 'box-x', 'index': 1}, 'max'),
    Output({'type': 'box-y', 'index': 1}, 'max'),
    Input('image-info', 'data')
)
def set_box_boundary(data):
    try:
        size = data['size']
    except TypeError:
        size = None
    return size[0], size[1], size[0], size[1]


if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)
