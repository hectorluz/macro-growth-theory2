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
x = 0.2
eta = 0.3
c = 0.4

index = pd.DataFrame(list(range(0,101)))

equal = ((d + n)/s)**(1/(a-1))

# Step 3. Create a plotly figure
trace_1 = go.Scatter(x = k[0], y = k[0] ** a,
                    name = 'f(x)',
                    line = dict(width = 2,
                                color = 'rgb(229, 151, 50)'))


layout = go.Layout(title = 'Produto, poupança, depreciação e crescimento populacional',
                    hovermode = 'closest',
                    )


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
                    html.P('d:     '),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('alpha:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                    
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
        dcc.Tab(label='Solow-Swan + Tech Progress', children=[
        html.Div([
            html.H1("Solow-Swan + Technological Progress")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d2', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n2', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s2', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('alpha:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a2', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('x:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='x2', type='number', value = x, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                    
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
        dcc.Tab(label='AK Model', children=[
        html.Div([
            html.H1("AK Model")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d3', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s3', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                                                
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
        dcc.Tab(label='R&D (Romer)', children=[
        html.Div([
            html.H1("R&D (Romer)")
                ]),
            html.Div([
                html.Div(
                    html.P('d:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='d4', type='number', value = d, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('n:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='n4', type='number', value = n, step=0.01),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('s:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='s4', type='number', value = s, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                                                
            html.Div([
                html.Div(
                    html.P('alpha:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='a4', type='number', value = a, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                    html.P('eta:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='eta4', type='number', value = eta, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                    
            html.Div([
                html.Div(
                    html.P('c:'),
                    style={'display': 'inline-block'}),
                html.Div(
                    dcc.Input(id='c4', type='number', value = c, step=0.05),
                    style={'display': 'inline-block'})],
                style={'width': '15%', 'display': 'inline-block'}),                    
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
                Output('plot4_3', 'figure'),
                Output('plot_4', 'figure'),
                Output('plot3_4', 'figure'),
                Output('plot2_4', 'figure'),
                Output('plot4_4', 'figure'),],
                [Input('d', 'value'), 
                Input('n', 'value'),
                Input('s', 'value'),
                Input('a', 'value'),
                Input('d2', 'value'), 
                Input('n2', 'value'),
                Input('s2', 'value'),
                Input('a2', 'value'),
                Input('x2', 'value'),
                Input('d3', 'value'),
                Input('s3', 'value'),
                Input('d4', 'value'),
                Input('n4', 'value'),
                Input('s4', 'value'),
                Input('a4', 'value'),
                Input('eta4', 'value'),
                Input('c4', 'value'),],
                )

def update_figure(input1,input2,input3,input4,input5,input6,input7,input8,
    input9,input10,input11,input12,input13,input14,input15,input16,input17,
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
                    xaxis_title="k",
                    )
    layout2 = go.Layout(title = 'gy, gk',
                    hovermode = 'closest',
                    xaxis_title="t",
                    )    
    layout3 = go.Layout(title = 'Taxas de crescimento',
                    hovermode = 'closest',
                    xaxis_title="k",
                    )
    layout4 = go.Layout(title = 'log(Y/L)',
                    hovermode = 'closest',
                    xaxis_title="t",
                    )

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
                    name='s * f(x)'))
        
    fig.add_trace(go.Scatter(x=k[0], y=(d+n)*(k[0]),
                    mode='lines',
                    name='(d + n) * k'))

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
    x2 = input9

    K = 1
    A_0 = 1
    L = 1


    for i in range(0,100):

        A = A_0 * np.exp(x2*i)
        Y = (K ** alpha2) * ((A * L) ** (1-alpha2))
        ytil = (Y/(A*L))
        ktil = (K/(A*L))    
        df = df.append(pd.DataFrame([s2,d2,Y,A,K,ytil,ktil]).transpose(),ignore_index=True)
        K = K + ((s2*Y) - (d2*K))
        L = L + (n2*L)
       

    df.columns = ['s','d','Y','A','K','ytil','ktil']

    layout5 = go.Layout(title = 'Produto, poupança, depreciação e crescimento populacional',
                    hovermode = 'closest',
                    xaxis_title="K",
                    )
    layout6 = go.Layout(title = 'gy, gk',
                    hovermode = 'closest',
                    xaxis_title="K",
                    xaxis_range=[1,3],
                    yaxis_range=[0,2],
                    )    
    layout7 = go.Layout(title = 'Taxas de crescimento',
                    hovermode = 'closest',
                    xaxis_title="¨k",
                    )
    layout8 = go.Layout(title = 'log(Y/L)',
                    hovermode = 'closest',
                    xaxis_title="¨k",
                    #xaxis_range=[0.45,0.5],
                    #yaxis_range=[0,5],                    
                    )
    
    trace_5 = go.Scatter(x = df['K'], y = df['Y'],
                        name = 'Y',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig5 = go.Figure(data = [trace_5],
        layout = layout5
        )

    fig5.add_trace(go.Scatter(x=df['K'], y=s*df['Y'],
                    mode='lines',
                    name='sY'))
        
    fig5.add_trace(go.Scatter(x=df['K'], y=(d2+n2)*(df['K']),
                    mode='lines',
                    name='(d+n)*K'))


    trace_6 = go.Scatter(x = df['K'], y = s2*df['A'],
                        name = 'sA',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig6 = go.Figure(data = [trace_6],
        layout = layout6
        )

    fig6.add_trace(go.Scatter(x=df['K'], y=df['d']+n,
                    mode='lines',
                    name='d + n'))

    trace_7 = go.Scatter(x = df['ktil'], y = df['ytil'],
                        name = 'ÿ',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig7 = go.Figure(data = [trace_7],
        layout = layout7
        )

    fig7.add_trace(go.Scatter(x=df['ktil'], y=s*df['ytil'],
                    mode='lines',
                    name='sÿ'))
        
    fig7.add_trace(go.Scatter(x=df['ktil'], y=(d2+n2+x2)*(df['ktil']),
                    mode='lines',
                    name='(d+n+x)*ÿ'))


    trace_8 = go.Scatter(x = df['ktil'], y = s2*df['A'],
                        name = 'sA',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig8 = go.Figure(data = [trace_8],
        layout = layout8
        )

    fig8.add_trace(go.Scatter(x=df['ktil'], y=df['d']+n,
                    mode='lines',
                    name='d+n'))
    #print('df:',df)

###
### AK Model
###


    d3 = input10
    s3 = input11
    

    n3 = 0

    K3 = 0.2
    A3 = 0.2
    X3 = A3 * K3

    df_3 = pd.DataFrame()

    for i in range(0,100):
        K3 = K3 + ((s3*X3) - (d3*K3))
        X3 = A3 * K3
        xtil3 = (X3/(A3))
        ktil3 = (K3/(A3))
        df_3 = df_3.append(pd.DataFrame([s3,d3,X3,A3,K3,xtil3,ktil3]).transpose(),ignore_index=True)

    df_3.columns = ['s','d','X','A','K','xtil','ktil']

    layout9 = go.Layout(title = 'Produto, poupança, depreciação e crescimento populacional',
                    hovermode = 'closest',
                    xaxis_title="K",
                    )
    layout10 = go.Layout(title = 'gy, gk',
                    hovermode = 'closest',
                    xaxis_title="K",
                    )    
    layout11 = go.Layout(title = 'Taxas de crescimento',
                    hovermode = 'closest',
                    xaxis_title="¨k",
                    )
    layout12 = go.Layout(title = 'log(Y/L)',
                    hovermode = 'closest',
                    xaxis_title="¨k",
                    )

    trace_9 = go.Scatter(x = df_3['X'], y = df_3['K'],
                        name = 'X',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig9 = go.Figure(data = [trace_9],
        layout = layout9
        )

    fig9.add_trace(go.Scatter(x=df_3['K'], y=s3*df_3['X'],
                    mode='lines',
                    name='sX'))
        
    fig9.add_trace(go.Scatter(x=df_3['K'], y=(d3+n3)*(df_3['K']),
                    mode='lines',
                    name='dX'))

    trace_10 = go.Scatter(x = df_3['K'], y = s3*df_3['A'],
                        name = 'sA',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig10 = go.Figure(data = [trace_10],
        layout = layout10
        )

    fig10.add_trace(go.Scatter(x=df_3['K'], y=df_3['d']+n3,
                    mode='lines',
                    name='d'))    

    trace_11 = go.Scatter(x = df_3['ktil'], y = df_3['xtil'],
                        name = '¨x',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig11 = go.Figure(data = [trace_11],
        layout = layout11
        )

    fig11.add_trace(go.Scatter(x=df_3['ktil'], y=s3*df_3['xtil'],
                    mode='lines',
                    name='s¨x'))
        
    fig11.add_trace(go.Scatter(x=df_3['ktil'], y=(d3+n3)*(df_3['xtil']),
                    mode='lines',
                    name='d¨x'))


    trace_12 = go.Scatter(x = df_3['ktil'], y = s3*df_3['A'],
                        name = 'sA',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig12 = go.Figure(data = [trace_12],
        layout = layout12
        )

    fig12.add_trace(go.Scatter(x=df_3['ktil'], y=df_3['d']+n3,
                    mode='lines',
                    name='d'))

    ###
    ### R&D
    ###

    d4 = input12
    n4 = input13
    s4 = input14
    alpha4 = input15
    eta4 = input16
    c4 = input17
  
    df_4 = pd.DataFrame()

    K = 0.01
    A = 0.2
    #alpha2 = 0.5
    Ly = 1
    Lr = 1
    L = Ly + Lr
    nr4 = n4
    #x2 = 0.02
    #s2 = 0.2
    #d2 = 0.2
    Y = (K ** alpha4) * ((A * Ly) ** (1-alpha4))

    for i in range(0,100):
        L = L + (n4*L)
        Lr = Lr + (nr4*Lr)
        Ly = L - Lr
        A = A + ((c4)*((A)**(eta4))*(Lr))
        K = K + ((s4*Y) - (d4*K))
        Y = (K ** alpha4) * ((A * Ly) ** (1-alpha4))
        gA2 = (((c4)*((A)**(eta4))*(Lr))/(A))
        ytil = (Y/(A*L))
        ktil = (K/(A*L))

        df_4 = df_4.append(pd.DataFrame([s4,d4,gA2,Y,A,K,ytil,ktil]).transpose(),ignore_index=True)

      
    df_4.columns = ['s','d','gA2','Y','A','K','ytil','ktil']

    g_df_4 = pd.DataFrame()

    for i in range(1,len(df_4)):
        gY = (df_4['Y'][i]-df_4['Y'][i-1])/df_4['Y'][i]
        gA = (df_4['A'][i]-df_4['A'][i-1])/df_4['A'][i]
        gK = (df_4['K'][i]-df_4['K'][i-1])/df_4['K'][i]

        g_df_4 = g_df_4.append(pd.DataFrame([gY,gA,gK]).transpose(),ignore_index=True)
    
    g_df_4.columns = ['gY','gA','gK']

    #print(g_df_4)

    layout13 = go.Layout(title = 'Produto, poupança, depreciação e crescimento populacional',
                    hovermode = 'closest',
                    xaxis_title="K",
                    )
    layout14 = go.Layout(title = 'gy, gk',
                    hovermode = 'closest',
                    xaxis_title="K",
                    )    
    layout15 = go.Layout(title = 'Taxas de crescimento',
                    hovermode = 'closest',
                    xaxis_title="¨k",
                    )
    layout16 = go.Layout(title = 'log(Y/L)',
                    hovermode = 'closest',
                    xaxis_title="¨k",
                    )


    trace_13 = go.Scatter(x = df_4['K'], y = df_4['Y'],
                        name = 'Y',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig13 = go.Figure(data = [trace_13],
        layout = layout13
        )

    fig13.add_trace(go.Scatter(x=df_4['K'], y=s4*df_4['Y'],
                    mode='lines',
                    name='sY'))
        
    fig13.add_trace(go.Scatter(x=df_4['K'], y=(d4+n4)*(df_4['K']),
                    mode='lines',
                    name='(d+n)K'))


    trace_14 = go.Scatter(x = df_4['K'], y = s4*df_4['A'],
                        name = 'sA',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig14 = go.Figure(data = [trace_14],
        layout = layout14
        )

    fig14.add_trace(go.Scatter(x=df_4['K'], y=df_4['d']+n4,
                    mode='lines',
                    name='d+n'))

    trace_15 = go.Scatter(x = df_4['ktil'], y = df_4['ytil'],
                        name = 'ÿ',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig15 = go.Figure(data = [trace_15],
        layout = layout15
        )

    fig15.add_trace(go.Scatter(x=df_4['ktil'], y=s4*df_4['ytil'],
                    mode='lines',
                    name='sÿ'))
        
    fig15.add_trace(go.Scatter(x=df_4['ktil'], y=(d4+n4+df_4['gA2'])*(df_4['ktil']),
                    mode='lines',
                    name='(d+n+ga)¨k'))


    trace_16 = go.Scatter(x = df_4['ktil'], y = s4*df_4['A'],
                        name = 'sA',
                        line = dict(width = 2,
                                    color = 'rgb(229, 151, 50)'))

    fig16 = go.Figure(data = [trace_16],
        layout = layout16
        )

    fig16.add_trace(go.Scatter(x=df_4['ktil'], y=df_4['d']+n4,
                    mode='lines',
                    name='d+n'))

    #print(df_4)

    return fig, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig16
    
server = app.server

    # Step 6. Add the server clause
if __name__ == '__main__':
    app.run_server()


