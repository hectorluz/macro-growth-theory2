import dash
import dash_core_components as dcc 
import dash_html_components as html 
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from dash.dependencies import Input, Output

# Step 1. Launch the application
app = dash.Dash()

# Step 2. Import the dataset
k = pd.DataFrame(list(np.arange(0,5,0.01)))

d = 0.2
n = 0.02
s = 0.3
a = 0.5

index = pd.DataFrame(list(range(0,101)))

equal = ((d + n)/s)**(1/(a-1))

# Step 3. Create a plotly figure
trace_1 = go.Scatter(x = k[0], y = k[0] ** a,
                    name = 'f(x)',
                    line = dict(width = 2,
                                color = 'rgb(229, 151, 50)'))


layout = go.Layout(title = 'Produto, poupança, depreciação e crescimento populacional',
                    hovermode = 'closest',
                    updatemenus=[dict(type="buttons",
                                    buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None])])
                                            ]                   )


fig = go.Figure(data = [trace_1], layout = layout)

fig.add_trace(go.Scatter(x=k[0], y=s*(k[0]**a),
                        mode='lines',
                        name='sf(x)'))

fig.add_trace(go.Scatter(x=k[0], y=(d+n)*(k[0]),
                        mode='lines',
                        name='df(x)'))

fig.layout.update(
    showlegend=False,
    annotations=[
        go.layout.Annotation(
            x=equal,
            y=0,
            xref="x",
            yref="y",
            text="Eq. inicial",
            arrowwidth=1,
            arrowhead=7,
            showarrow=True,
            ax=0,
            ay=-200
        )
    ]
)


trace_2 = go.Scatter(x = [0], y = [0],
                        name = 'gx',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))


fig2 = go.Figure(data = [trace_2], layout = layout)

#fig3.add_trace(go.Scatter(x=k[0], y=(s*(k[0]**a))/k[0],
#                        mode='lines',
#                        name='sf(x)'))


trace_3 = go.Scatter(x=k[0], y=(s*(k[0]**a))/k[0],
                        name = 'gx',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

fig3 = go.Figure(data = [trace_3], layout = layout)


fig3.add_trace(go.Scatter(x=k[0], y=[d+n]*len(k[0]),
                        mode='lines',
                        name='df(x)'))



    # Step 4. Create a Dash layout
app.layout = html.Div([
        # a header and a paragraph
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Solow-Swan Model', children=[
        html.Div([
            html.H1("Solow-Swan Model")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('a:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                    
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot', figure = fig)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot2', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ]),
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot3', figure = fig3)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot4', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ])
        ]),
        dcc.Tab(label='Solow + Exogenous', children=[
        html.Div([
            html.H1("Solow-Swan Model")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d2', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n2', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s2', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('a:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a2', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                    
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot_2', figure = fig)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot2_2', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ]),
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot3_2', figure = fig3)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot4_2', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ])
        ]),
        dcc.Tab(label='Modelo AK', children=[
        html.Div([
            html.H1("Modelo AK")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d3', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n3', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s3', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('a:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a3', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                    
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot_3', figure = fig)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot2_3', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ]),
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot3_3', figure = fig3)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot4_3', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ])
        ]),
        dcc.Tab(label='Endogenous Growth', children=[
        html.Div([
            html.H1("Solow-Swan Model")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d4', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n4', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s4', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('a:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a4', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '12.5%', 'display': 'inline-block'}),                    
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot_4', figure = fig)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot2_4', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ]),
        html.Div([
            html.Div([
                dcc.Graph(id = 'plot3_4', figure = fig3)
                    ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id = 'plot4_4', figure = fig2)],
                style={'width': '48%', 'float':'right', 'display': 'inline-block'})
                    ])
        ]),
    ])
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

    # Step 5. Add callback functions
@app.callback([Output('plot', 'figure'),
                Output('plot2', 'figure'),
                Output('plot3', 'figure'),
                Output('plot4', 'figure'),
                Output('plot_2', 'figure'),
                Output('plot3_2', 'figure'),
                Output('plot2_2', 'figure'),
                Output('plot4_2', 'figure'),                
                Output('plot_3', 'figure'),
                Output('plot3_3', 'figure'),
                Output('plot2_3', 'figure'),
                Output('plot4_3', 'figure'),],
                [Input('d', 'value'), 
                Input('n', 'value'),
                Input('s', 'value'),
                Input('a', 'value'),
                Input('d2', 'value'), 
                Input('n2', 'value'),
                Input('s2', 'value'),
                Input('a2', 'value'),
                Input('s3', 'value'),
                Input('d3', 'value'),
                Input('s4', 'value'),
                Input('d4', 'value')],
                )

def update_figure(input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,
    input11,input12,
):
        # updating the plot

###
### SOLOW-SWAN MODEL
###

    d = input1
    n = input2
    s = input3
    a = input4

    new_equal = ((d + n)/s)**(1/(a-1))
            
    trace_1 = go.Scatter(x = k[0], y = k[0] ** a,
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))
        
    layout1 = go.Layout(title = 'Produto, poupança, depreciação e crescimento populacional',
                    hovermode = 'closest',
                    updatemenus=[
                            dict(type="buttons",
                            buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])
                                        ])
    layout2 = go.Layout(title = 'gy, gk',
                    hovermode = 'closest',
                    updatemenus=[
                            dict(type="buttons",
                            buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])
                                        ])
        
    layout3 = go.Layout(title = 'Taxas de crescimento',
                    hovermode = 'closest',
                    updatemenus=[
                            dict(type="buttons",
                            buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])
                                        ])

    layout4 = go.Layout(title = 'log(Y/L)',
                    hovermode = 'closest',
                    updatemenus=[
                            dict(type="buttons",
                            buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])
                                        ])

    kz = equal
    yz = kz**a
    gk = (s*(kz**(a-1)))-d-n
    gy = 0

    i = 0

    time = pd.DataFrame(np.matrix([0,kz,yz,gk]))

    for i in range(0,201):
        gk = round(s*(time.iloc[i,1]**(a-1))-d-n,5)
        kz = time.iloc[i,1] + gk
        yz = time.iloc[i,1]**a
        gy = round(yz - time.iloc[i-1,2],5)
        #print(gy)
        #print(np.log(yz))
        time = time.append(pd.DataFrame(np.matrix([i,kz,yz,gk,gy])))

    time.columns = ['idx','kz','yz','gk','gy']

    fig = go.Figure(data = [trace_1],
        layout = layout1
        )
                
    trace_2 = go.Scatter(x = time.iloc[1:,0], y = time.iloc[1:,3],
                        name = 'gk',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))


    fig2 = go.Figure(data = [trace_2],
        layout = layout2
        )    

    fig2.add_trace(go.Scatter(x=time.iloc[1:,0], y=time.iloc[3:,4],
                    mode='lines',
                    name='gy',
                    line = dict(color = 'blue')))    

    trace_3 = go.Scatter(x=k[0], y=(s*(k[0]**a))/k[0],
                            name = 'sf(k)/k',
                            line = dict(width = 2,
                                        color = 'rgb(229, 151, 50)'))

    fig3 = go.Figure(data = [trace_3], layout = layout3)


    fig3.add_trace(go.Scatter(x=k[0], y=[d+n]*len(k[0]),
                            mode='lines',
                            name='d + n',
                            line = dict(color = 'blue')                            ))
    
    trace_4 = go.Scatter(x=time.iloc[1:,0], y=np.log(time.iloc[3:,2]),
                            name = 'gx',
                            line = dict(width = 2,
                                        color = 'rgb(229, 151, 50)'))

    fig4 = go.Figure(data = [trace_4], layout = layout4)


    fig.add_trace(go.Scatter(x=k[0], y=s*(k[0]**a),
                    mode='lines',
                    name='sf(x)'))
        
    fig.add_trace(go.Scatter(x=k[0], y=(d+n)*(k[0]),
                    mode='lines',
                    name='df(x)'))

    fig.layout.update(
        showlegend=False,
        annotations=[
            go.layout.Annotation(
                x=equal,
                y=0,
                xref="x",
                yref="y",
                text="Eq. inicial",
                arrowwidth=1,
                arrowhead=7,
                showarrow=True,
                ax=0,
                ay=-200
            ),
            go.layout.Annotation(
                x=new_equal,
                y=0,
                xref="x",
                yref="y",
                text="Novo eq.",
                showarrow=True,
                arrowwidth=1,
                arrowhead=7,
                ax=0,
                ay=-230
            )
        ]
    )


###
### Solow + exogenous
###
    
    df = pd.DataFrame()

    d2 = input5
    n2 = input6
    s2 = input7
    alpha2 = input8

    K = 1
    A_0 = 1
    #alpha2 = 0.5
    L = 1
    x2 = 0.02
    #s2 = 0.2
    #d2 = 0.2
    A = A_0 * np.exp(x2*i)
    Y = (K ** alpha2) * ((A * L) ** (1-alpha2))

    for i in range(0,100):
        A = A_0 * np.exp(x2*i)
        K = K + ((s2*Y) - (d2*K))
        L = L + (n2*L)
        Y = (K ** alpha2) * ((A * L) ** (1-alpha2))
        ytil = (Y/(A*L))
        ktil = (K/(A*L))
        df = df.append(pd.DataFrame([s2,d2,Y,A,K,ytil,ktil]).transpose(),ignore_index=True)

    df.columns = ['s','d','Y','A','K','ytil','ktil']
    
    trace_5 = go.Scatter(x = df['K'], y = df['Y'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig5 = go.Figure(data = [trace_5],
        layout = layout1
        )

    fig5.add_trace(go.Scatter(x=df['K'], y=s*df['Y'],
                    mode='lines',
                    name='sf(x)'))
        
    fig5.add_trace(go.Scatter(x=df['K'], y=(d2+n2)*(df['K']),
                    mode='lines',
                    name='df(x)'))


    trace_6 = go.Scatter(x = df['K'], y = s2*df['A'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig6 = go.Figure(data = [trace_6],
        layout = layout1
        )

    fig6.add_trace(go.Scatter(x=df['K'], y=df['d']+n,
                    mode='lines',
                    name='df(x)'))

    trace_7 = go.Scatter(x = df['ktil'], y = df['ytil'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig7 = go.Figure(data = [trace_7],
        layout = layout1
        )

    fig7.add_trace(go.Scatter(x=df['ktil'], y=s*df['ytil'],
                    mode='lines',
                    name='sf(x)'))
        
    fig7.add_trace(go.Scatter(x=df['ktil'], y=(d2+n2)*(df['ktil']),
                    mode='lines',
                    name='df(x)'))


    trace_8 = go.Scatter(x = df['ktil'], y = s2*df['A'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig8 = go.Figure(data = [trace_8],
        layout = layout1
        )

    fig8.add_trace(go.Scatter(x=df['ktil'], y=df['d']+n,
                    mode='lines',
                    name='df(x)'))

###
### AK Model
###


    s3 = input9
    d3 = input10

    n3 = 0

    K3 = 1
    A3 = 1
    X3 = A3 * K3

    df_3 = pd.DataFrame()

    for i in range(0,100):
        K3 = K3 + ((s3*X3) - (d3*K3))
        X3 = A3 * K3
        xtil3 = (X3/(A3))
        ktil3 = (K3/(A3))
        df_3 = df_3.append(pd.DataFrame([s3,d3,X3,A3,K3,xtil3,ktil3]).transpose(),ignore_index=True)

    df_3.columns = ['s','d','X','A','K','xtil','ktil']

    trace_9 = go.Scatter(x = df_3['X'], y = df_3['K'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig9 = go.Figure(data = [trace_9],
        layout = layout1
        )

    fig9.add_trace(go.Scatter(x=df_3['K'], y=s3*df_3['X'],
                    mode='lines',
                    name='sf(x)'))
        
    fig9.add_trace(go.Scatter(x=df_3['K'], y=(d3+n3)*(df_3['K']),
                    mode='lines',
                    name='df(x)'))

    trace_10 = go.Scatter(x = df_3['K'], y = s3*df_3['A'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig10 = go.Figure(data = [trace_10],
        layout = layout1
        )

    fig10.add_trace(go.Scatter(x=df_3['K'], y=df_3['d']+n,
                    mode='lines',
                    name='df(x)'))    

    trace_11 = go.Scatter(x = df_3['ktil'], y = df_3['xtil'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig11 = go.Figure(data = [trace_11],
        layout = layout1
        )

    fig11.add_trace(go.Scatter(x=df_3['ktil'], y=s3*df_3['xtil'],
                    mode='lines',
                    name='sf(x)'))
        
    fig11.add_trace(go.Scatter(x=df_3['ktil'], y=(d3+n3)*(df_3['xtil']),
                    mode='lines',
                    name='df(x)'))


    trace_12 = go.Scatter(x = df_3['ktil'], y = s3*df_3['A'],
                        name = 'f(x)',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig12 = go.Figure(data = [trace_12],
        layout = layout1
        )

    fig12.add_trace(go.Scatter(x=df_3['ktil'], y=df_3['d']+n3,
                    mode='lines',
                    name='df(x)'))

    print(df_3)

    return fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12
    
server = app.server

    # Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server()


