import dash_bootstrap_components as dbc


def GroupInput(index,
               size=None,
               labeled=False,
               placeholder_x='x coordinate',
               placeholder_y='y coordinate'
               ):
    if labeled:
        input_type = 'point-'
    else:
        input_type = 'box-'

    content = [
        dbc.Col([
            dbc.Label(f'x{index}'),
            dbc.Input(
                id={'type': input_type + 'x', 'index': index},
                type='number',
                min=0,
                max=size[0] if size else None,
                placeholder=placeholder_x
            )
        ]),
        dbc.Col([
            dbc.Label(f'y{index}'),
            dbc.Input(
                id={'type': input_type + 'y', 'index': index},
                type='number',
                min=0,
                max=size[1] if size else None,
                placeholder=placeholder_y
            )
        ])
    ]

    if labeled:

        content.append(
            dbc.Col([
                dbc.Label('Inside'),
                dbc.Switch(id={'type': 'label', 'index': index}, value=True)
            ], width=2)
        )

    return dbc.Row(content)
