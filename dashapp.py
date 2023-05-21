import datetime

import numpy as np
import plotly.express as px
import dash
import torch.cuda
from dash import dcc, html
from dash import Input, Output, State, Patch, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import dash_resusable_components as drc
from sam_tools import *
from image_tools import *

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
        html.H5('Uploaded image will be showed below')
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
            },
            responsive='auto'
        )
    ], style={'display': 'flex', 'justify-content': 'center'}),
    dcc.Store(id='image-info', storage_type='memory')

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
        ], width=4),

        dbc.Col([
            dbc.Label('Device'),
            dbc.RadioItems(
                id='device-selection',
                options={'auto': 'Auto', 'cpu': 'CPU', 'cuda': 'CUDA'},
                value='auto'
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Button('Can I use CUDA?', id='cuda-check', color='warning', class_name='mt-3')
                ], width=6),
                dbc.Col([
                    dbc.Alert(id='cuda-check-result', is_open=False, dismissable=True, fade=True, duration=3000)
                ], width=6)

            ], style={'height': '5em'})
        ], width=8)
    ])
])

point_input_area = dbc.Collapse([
    dbc.Button('Add Point', id='add-point', color='secondary', class_name='mt-3'),
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
        dbc.Button('Segment', id='segment', color='primary', class_name='mt-3',
                   style={'width': '100%'})
    ], width=6),
    dbc.Col([
        dbc.Button('View Prompts', id='view-prompts', color='info', class_name='mt-3',
                   style={'width': '100%'})
    ], width=3),
    dbc.Col([
        dbc.Button('Clear Prompts', id='clear-prompts', color='secondary', class_name='mt-3',
                   style={'width': '100%'})
    ], width=3)
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
    html.Br(),
    html.Br(),
    dcc.Loading([
        html.Div(id='sam-info', style={'visibility': 'hidden'})
    ])

], style={'margin-right': '10px'})

output_image = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Label('Mask:')
        ], lg=1, md=2, class_name='mr-0'),
        dbc.Col([
            dbc.RadioItems(
                id='mask',
                options=[0, 1, 2],
                value=0,
                inline=True)
        ], class_name='ml-0')
    ]),

    dbc.Row([
        dcc.Loading([
            dcc.Graph(
                id='output-image',
                figure={
                    'layout': {
                        'height': 700,
                        'width': 700,
                        'margin': {'l': 0, 'b': 0, 't': 0, 'r': 0},
                        'xaxis': {'showgrid': False, 'showticklabels': False, 'visible': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'visible': False}
                    }
                },
                config={
                    'displaylogo': False,
                    'scrollZoom': True,
                    'staticPlot': True
                }
            )
        ]),
        dcc.Store(id='output-image-data', storage_type='memory')
    ])
])

output_settings = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Label('Format:')
        ], lg=1, md=2, class_name='mr-0'),

        dbc.Col([
            dbc.RadioItems(
                id='format',
                options=['png', 'jpg'],
                value='png',
                inline=True
            )
        ], class_name='ml-0'),

        dbc.Col([
            dbc.Label('Output Mode:'),
        ], lg=2, md=4, class_name='mr-0'),
        dbc.Col([
            dbc.Checklist(
                id='output-mode',
                options=['cut-out', 'mask only'],
                value=['cut-out', 'mask only'],
                inline=True
            )
        ])
    ]),

    dbc.Row([
        dbc.Button('Export', id='export', color='success', class_name='mt-3')
    ]),

    dbc.Row([
        html.Div(id='export-info', style={'visibility': 'hidden'})
    ])
], style={'margin-right': '10px'})

output_area = dbc.Collapse([
    dbc.Row([
        dbc.Col([
            output_image
        ], lg=6, md=12),
        html.Br(),

        dbc.Col([
            output_settings
        ], lg=6, md=12)
    ])
], id='output-area', is_open=True)

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
    html.Br(),

    dbc.Row([
        output_area
    ])

], fluid=True)


@app.callback(
    Output('image-info', 'data'),
    Output('image-title', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('upload-image', 'last_modified')
)
def upload_image(content, name, date):
    if content is None:
        raise PreventUpdate
    image, filename = parse_contents(content, name)
    size = image.size
    img_key = 'image'
    img_data = np.asarray(image)
    save_temp_data(img_data, img_key)
    info = [
        html.H5(f'uploaded {size[0]} × {size[1]} image'),
        html.H6(f'filename: {filename}'),
        html.H6(f'last modified: {datetime.datetime.fromtimestamp(date)}')
    ]
    image = load_temp_data('image')

    return {'image': img_key, 'size': size, 'name': name}, info


@app.callback(
    Output('image', 'figure'),
    Input('image-info', 'data'),
    Input('view-prompts', 'n_clicks'),
    Input('clear-prompts', 'n_clicks'),
    State({'type': 'point-x', 'index': ALL}, 'value'),
    State({'type': 'point-y', 'index': ALL}, 'value'),
    State({'type': 'label', 'index': ALL}, 'value'),
    State({'type': 'box-x', 'index': 0}, 'value'),
    State({'type': 'box-y', 'index': 0}, 'value'),
    State({'type': 'box-x', 'index': 1}, 'value'),
    State({'type': 'box-y', 'index': 1}, 'value'),
    State('mode', 'active_tab'),
)
def draw_image(data, view_clicks, clear_clicks, point_xs, point_ys, labels, box_x0, box_y0, box_x1, box_y1, mode):
    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] == 'view-prompts.n_clicks':
        match mode:
            case 'point':
                no_point = (len(point_xs) == 0) or (len(point_ys) == 0)
                missing_point = not all(point_xs) or not all(point_ys)
                if no_point or missing_point:
                    raise PreventUpdate
                points = np.array(list(zip(point_xs, point_ys)))
                labels = np.array(labels)
                try:
                    img_data = load_temp_data(data['image'])
                    image = Image.fromarray(np.array(img_data, dtype=np.uint8))
                except TypeError:
                    raise PreventUpdate
                new_img = draw_points(image, points, labels)
                fig = px.imshow(new_img)
            case 'box':
                missing_vertex = not (box_x0 and box_y0 and box_x1 and box_y1)
                if missing_vertex:
                    raise PreventUpdate
                box = [box_x0, box_y0, box_x1, box_y1]
                try:
                    img_data = load_temp_data(data['image'])
                    image = Image.fromarray(np.array(img_data, dtype=np.uint8))
                except TypeError:
                    raise PreventUpdate
                new_img = draw_box(image, box)
                fig = px.imshow(new_img)
            case _:
                raise PreventUpdate
    else:
        try:
            img_data = load_temp_data(data['image'])
            image = Image.fromarray(np.array(img_data, dtype=np.uint8))
        except TypeError:
            raise PreventUpdate
        fig = px.imshow(image)

    fig.update_traces(hovertemplate='    (%{x}, %{y})<extra></extra>')
    fig.update_layout(
        width=800, height=800,
        margin=dict(l=10, r=50, b=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor='white',
            font_size=16,
            font_family='Rockwell',
            align='right',
            namelength=16
        )
    )

    return fig


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
    if not data:
        raise PreventUpdate

    try:
        size = data['size']
    except TypeError:
        size = None

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


@app.callback(
    Output('cuda-check-result', 'children'),
    Output('cuda-check-result', 'color'),
    Output('cuda-check-result', 'is_open'),
    Input('cuda-check', 'n_clicks')
)
def check_cuda(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            return 'CUDA is available', 'success', True
        else:
            return 'CUDA is not available', 'danger', True


@app.callback(
    Output('output-image-data', 'data'),
    Output('sam-info', 'children'),
    Output('sam-info', 'style'),
    Input('segment', 'n_clicks'),
    Input('upload-image', 'contents'),
    State('image-info', 'data'),
    State('mode', 'active_tab'),
    State('model-selection', 'value'),
    State('device-selection', 'value'),
    State({'type': 'point-x', 'index': ALL}, 'value'),
    State({'type': 'point-y', 'index': ALL}, 'value'),
    State({'type': 'label', 'index': ALL}, 'value'),
    State({'type': 'box-x', 'index': 0}, 'value'),
    State({'type': 'box-y', 'index': 0}, 'value'),
    State({'type': 'box-x', 'index': 1}, 'value'),
    State({'type': 'box-y', 'index': 1}, 'value'),
)
def segment(n_clicks, upload, data, mode, model, device, point_xs, point_ys, labels, box_x0, box_y0, box_x1, box_y1):
    if not n_clicks:
        raise PreventUpdate

    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'upload-image.contents':
        return None, None, {'visiblity': 'hidden'}

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    match mode:
        case 'point':
            no_point = (len(point_xs) == 0) or (len(point_ys) == 0)
            missing_point = not all(point_xs) or not all(point_ys)
            if no_point or missing_point:
                raise PreventUpdate
            points = np.array(list(zip(point_xs, point_ys)))
            labels = np.array(labels)

            try:
                img_data = load_temp_data(data['image'])
                array = np.array(img_data, dtype=np.uint8)
                image = Image.fromarray(array)
                image = image.convert('RGB')
                array = np.asarray(image)
            except TypeError:
                raise PreventUpdate
            try:
                masks, _, _, info = seg_with_points(array, points, labels,
                                                    model_type=model, device=device, need_info=True)
                mask_key = 'masks'
                save_temp_data(masks, mask_key)
                return {'masks': mask_key, 'mode': mode}, info, {'visibility': 'visible'}
            except torch.cuda.OutOfMemoryError:
                return None, 'CUDA out of memory', {'visibility': 'visible'}

        case 'box':
            missing_vertex = not (box_x0 and box_y0 and box_x1 and box_y1)
            if missing_vertex:
                raise PreventUpdate
            box = np.array([box_x0, box_y0, box_x1, box_y1])
            try:
                img_data = load_temp_data(data['image'])
                array = np.array(img_data, dtype=np.uint8)
                image = Image.fromarray(array)
                image = image.convert('RGB')
                array = np.asarray(image)
            except TypeError:
                raise PreventUpdate
            try:
                masks, _, _, info = seg_with_box(array, box, model_type=model, device=device, need_info=True)
                mask_key = 'masks'
                save_temp_data(masks, mask_key)
                return {'masks': mask_key, 'mode': mode}, info, {'visibility': 'visible'}
            except torch.cuda.OutOfMemoryError:
                return None, 'CUDA out of memory', {'visibility': 'visible'}

        case 'auto':
            try:
                img_data = load_temp_data(data['image'])
                array = np.array(img_data, dtype=np.uint8)
                image = Image.fromarray(array)
                image = image.convert('RGB')
                array = np.asarray(image)
            except TypeError:
                raise PreventUpdate
            try:
                masks, info = auto_seg(array, model_type=model, device=device, need_info=True)
                mask_key = 'masks'
                save_temp_data(masks, mask_key)
                return {'masks': mask_key, 'mode': mode}, info, {'visibility': 'visible'}
            except torch.cuda.OutOfMemoryError:
                return None, 'CUDA out of memory', {'visibility': 'visible'}


@app.callback(
    Output('output-image', 'figure'),
    Input('output-image-data', 'data'),
    Input('mask', 'value'),
    State('image-info', 'data')
)
def draw_result(mask_data, mask_number, img_info):
    try:
        masks = load_temp_data(mask_data['masks'])
        mode = mask_data['mode']
        img_data = load_temp_data(img_info['image'])
    except TypeError:
        raise PreventUpdate

    ctx = dash.callback_context
    if np.shape(masks)[0] == 0:
        raise PreventUpdate
    if ctx.triggered[0]['prop_id'] == 'mask.value':
        if mode == 'auto':
            raise PreventUpdate
        image = Image.fromarray(np.array(img_data, dtype=np.uint8))
        mask = [masks[mask_number]]
        masked_image = draw_mask(image, mask)
        fig = px.imshow(masked_image)
        fig.update_layout(
            width=800, height=800
        )
        return fig
    else:
        match mode:
            case 'point':
                image = Image.fromarray(np.array(img_data, dtype=np.uint8))
                mask = [masks[mask_number]]
                masked_image = draw_mask(image, mask)
                fig = px.imshow(masked_image)
                fig.update_layout(
                    width=800, height=800
                )
                return fig

            case 'box':
                image = Image.fromarray(np.array(img_data, dtype=np.uint8))
                mask = [masks[mask_number]]
                masked_image = draw_mask(image, mask)
                fig = px.imshow(masked_image)
                fig.update_layout(
                    width=800, height=800
                )
                return fig

            case 'auto':
                image = Image.fromarray(np.array(img_data, dtype=np.uint8))
                extracted_masks = [mask['segmentation'] for mask in masks]
                image = draw_mask(image, extracted_masks)
                fig = px.imshow(image)
                fig.update_layout(
                    width=800, height=800
                )
                return fig


@app.callback(
    Output('export-info', 'children'),
    Output('export-info', 'style'),
    Input('export', 'n_clicks'),
    State('format', 'value'),
    State('output-mode', 'value'),
    State('image-info', 'data'),
    State('output-image-data', 'data')
)
def export_image(n_clicks, format, output_mode, image_data, mask_data):
    if n_clicks is None:
        raise PreventUpdate
    if not output_mode:
        raise PreventUpdate
    try:
        img_data = load_temp_data(image_data['image'])
        img_name = image_data['name']
        image = Image.fromarray(np.array(img_data, dtype=np.uint8))
        masks = load_temp_data(mask_data['masks'])
        seg_mode = mask_data['mode']
    except TypeError:
        raise PreventUpdate

    output_folder = 'output/' + img_name + '_seg'

    if 'cut-out' in output_mode:
        for i in range(len(masks)):
            if seg_mode == 'auto':
                layer = mask_to_layer(image, masks[i]['segmentation'], 'crop')
            else:
                layer = mask_to_layer(image, masks[i], 'crop')
            if format == 'jpg':
                layer = layer.convert('RGB')
            name = seg_mode + '_crop_' + str(i) + '.' + format

            export_image_to_file(layer, name, output_folder)

    if 'mask only' in output_mode:
        for i in range(len(masks)):
            if seg_mode == 'auto':
                layer = mask_to_layer(image, masks[i]['segmentation'], 'mask')
            else:
                layer = mask_to_layer(image, masks[i], 'mask')
            if format == 'jpg':
                layer = layer.convert('RGB')
            name = seg_mode + '_mask_' + str(i) + '.' + format
            export_image_to_file(layer, name, output_folder)

    return f'Exported {len(masks)} masks, folder: {output_folder}', {'visibility': 'visible'}


if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=False, port=8050, host='')
