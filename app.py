import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from datetime import datetime

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from icecream import ic

import requests

from colors import estatus_colors
from colors import colors

url = "http://127.0.0.1:5000"
url = "http://miros.geovirtual.mx:5000"

months = {    
    1: "ENERO",
    2: "FEBRERO",
    3: "MARZO",
    4: "ABRIL",
    5: "MAYO",
    6: "JUNIO",
    7: "JULIO",
    8: "AGOSTO",
    9: "SEPTIEMBRE",
    10: "OCTUBRE",
    11: "NOVIEMBRE",
    12: "DICIEMBRE",
}

ic()
## --------------------------- READ DATA FROM DB ----------------------
data = requests.get(f"{url}/data/sm")
df = pd.read_json(data.text)
tipo_servicio = pd.DataFrame.from_dict(requests.get(f"{url}/robot/catalogos/tipo_sm").json()['data'])
tipo_servicio = tipo_servicio.iloc[:, [0,2]]

sub_tipo_servicio = pd.DataFrame.from_dict(requests.get(f"{url}/robot/catalogos/subtipo_sm").json()['data'])
sub_tipo_servicio = sub_tipo_servicio.iloc[:, [2, 1]]
## --------------------------- READ DATA FROM DB ----------------------

## --------------------------- FILTERIONG BY ZAPOPAN ----------------------

df['SM_FECHA_REGISTRO'] = pd.to_datetime(df['SM_FECHA_REGISTRO'], format='%Y-%m-%d', errors='coerce')
df['SM_FECHA_CIERRE'] = pd.to_datetime(df['SM_FECHA_CIERRE'], format='%Y-%m-%d', errors='coerce')
df_zapopan = df.query(""" NOM_MUN == 'ZAPOPAN' """).merge(tipo_servicio).merge(sub_tipo_servicio)
df_zapopan.TS_TIPO_SERVICIO.replace({
    'Agua potable y Saneamiento (drenaje, alcantarillado y tratamiento de aguas residuales)': 'Agua potable y Saneamiento'
}, inplace=True)
df_zapopan['ESTATUS'] = np.where(
    df_zapopan['SM_ESTATUS'] == 0, "SIN ATENDER",
        np.where(
            df_zapopan['SM_ESTATUS'] == 1, "ATENDIDO", "EN PROCESO"
        )
)
## CHANGIN SATUS IDX BY LABEL

## --------------------------- FILTERIONG BY ZAPOPAN ----------------------
today = date.today()
today_iso = date.today().isocalendar()
df_zapopan = df_zapopan[df_zapopan.SM_FECHA_REGISTRO.dt.date < today + timedelta(days=1)]  ## FILTERING DATA LESS THAN TODAY

ic()
## --------------------------- DATE FILTERS ----------------------
current_week_filter = df_zapopan.SM_FECHA_REGISTRO.dt.isocalendar()['week'] == today_iso[1]

last_week_filter = df_zapopan.SM_FECHA_REGISTRO.dt.isocalendar()['week'] == today_iso[1] - 1

current_month_filter = df_zapopan.SM_FECHA_REGISTRO.dt.month == today.month

last_month_filter = df_zapopan.SM_FECHA_REGISTRO.dt.month == (today.month - 1)

today_filter = df_zapopan.SM_FECHA_REGISTRO.dt.date == today

yesterday_filter = df_zapopan.SM_FECHA_REGISTRO.dt.date == today - timedelta(days=1)
## --------------------------- DATE FILTERS ----------------------



ic()
## --------------------------- DIFF MONTHS ----------------------
diff_months = df_zapopan[current_month_filter].shape[0] - df_zapopan[last_month_filter].shape[0]
increase_month = f"Hay {abs(diff_months)} reportes más que el mes pasado, lo que indica un crecimiento del {abs(diff_months)/df_zapopan[last_month_filter].shape[0]*100:.1f}% respecto al mes anterior"
decrease_month = f"Hay {abs(diff_months)} reportes menos que el mes pasado, lo que indica {abs(diff_months)/df_zapopan[last_month_filter].shape[0]*100:.1f}% menos que el mes anterior"
increase_month if diff_months > 0 else decrease_month
## --------------------------- DIFF MONTHS ----------------------

## --------------------------- YESTERDAY ----------------------
yesterday = df_zapopan[yesterday_filter]
yesterday_atendidos =  yesterday.query("ESTATUS == 'ATENDIDO'")
atendidos_msg_yesterday = 'todos' if yesterday.shape[0] == yesterday_atendidos.shape[0] else yesterday_atendidos.shape[0]
f"El día de ayer hubo {yesterday.shape[0]} reportes, de los cuales {atendidos_msg_yesterday} han sido atendidos"
## --------------------------- YESTERDAY ----------------------

## --------------------------- DIFF WEEK ----------------------
last_week = df_zapopan[last_week_filter]
last_week_atendidos =  last_week.query("ESTATUS == 'ATENDIDO'")
atendidos_msg_week =     'todos' if last_week.shape[0] == last_week_atendidos.shape[0] else last_week_atendidos.shape[0]
f"La semana pasada hubo {last_week.shape[0]} reportes, de los cuales {atendidos_msg_week} han sido atendidos"
## --------------------------- DIFF WEEK ----------------------

ic()
## --------------------------- TOTALS ----------------------
total = df_zapopan.shape[0]
atendidos = df_zapopan.query("ESTATUS == 'ATENDIDO'").shape[0]
no_atendidos = df_zapopan.query("ESTATUS == 'SIN ATENDER'").shape[0]
en_proceso = df_zapopan.query("ESTATUS == 'EN PROCESO'").shape[0]
total, atendidos, no_atendidos, en_proceso

this_month = df_zapopan[current_month_filter]
total_this_month = this_month.shape[0]
atendidos_this_month = this_month.query("ESTATUS == 'ATENDIDO'").shape[0]
no_atendidos_this_month = this_month.query("ESTATUS == 'SIN ATENDER'").shape[0]
en_proceso_this_month = this_month.query("ESTATUS == 'EN PROCESO'").shape[0]
total_this_month, atendidos_this_month, no_atendidos_this_month, en_proceso_this_month

today_ = df_zapopan[today_filter]
total_today_ = today_.shape[0]
atendidos_today_ = today_.query("ESTATUS == 'ATENDIDO'").shape[0]
no_atendidos_today_ = today_.query("ESTATUS == 'SIN ATENDER'").shape[0]
en_proceso_today_ = today_.query("ESTATUS == 'EN PROCESO'").shape[0]
total_today_, atendidos_today_, no_atendidos_today_, en_proceso_today_

## --------------------------- TOTALS ----------------------

## --------------------------- STATUS BY COLONIA ----------------------
temp = df_zapopan.groupby(
    ["SM_FECHA_REGISTRO", "TS_TIPO_SERVICIO", "STS_SUBTIPO_SERVICIO", 
     "NOM_COL", "ESTATUS"])['SERV_MUNICIPAL_ID'].count().reset_index()
temp.columns = ("FECHA", "TIPO DE SERVICIO", "SUBTIPO DE SERVICIO", "COLONIA", "ESTATUS", "REPORTES")
temp['MES'] = temp.FECHA.apply(lambda item: months[item.month])
fig_bar_status_by_tipo = px.bar(
    temp, x="TIPO DE SERVICIO", y="REPORTES", color="ESTATUS", barmode="group",
    color_discrete_map=estatus_colors
)
fig_bar_status_by_tipo.update_layout(
    font_color=colors['title-dark'],
    paper_bgcolor=colors['background-figures-dark'],
    plot_bgcolor=colors['background-figures-dark'])
## --------------------------- STATUS BY COLONIA ----------------------

## --------------------------- STATUS PER ... ----------------------
fig_pie_estatus_total = px.pie(temp, names="ESTATUS", color="ESTATUS", values="REPORTES", 
             labels="ESTATUS",
             color_discrete_map=estatus_colors)
fig_pie_estatus_total.update_traces(textposition='auto', textinfo='percent+label')
fig_pie_estatus_total.update_layout(font_color=colors['title-dark'], title_font_color=colors['title-dark'])

filter_ = temp.FECHA.dt.month == today.month
fig_pie_estatus_this_month = px.pie(temp[filter_], names="ESTATUS", color="ESTATUS", values="REPORTES", 
             labels="ESTATUS",
             color_discrete_map=estatus_colors)
fig_pie_estatus_this_month.update_traces(textposition='auto', textinfo='percent+label')
fig_pie_estatus_this_month.update_layout(font_color=colors['title-dark'], title_font_color=colors['title-dark'])

filter_ = temp.FECHA.dt.isocalendar()['week'] == today_iso[1]
fig_pie_estatus_this_week = px.pie(temp[filter_], names="ESTATUS", color="ESTATUS", values="REPORTES", 
             labels="ESTATUS",
             color_discrete_map=estatus_colors)
fig_pie_estatus_this_week.update_traces(textposition='auto', textinfo='percent+label')
fig_pie_estatus_this_week.update_layout(font_color=colors['title-dark'], title_font_color=colors['title-dark'])

## --------------------------- STATUS PER ... ----------------------

## --------------------------- TIME IN CLOSING REPORTS----------------------
diff_cierre = df_zapopan.DIF_FECHA.mean()
msg_diff_cierre = f'## En promedio tardan **{diff_cierre: .2f}** días en cerrar un reporte'

tmp_ = df_zapopan.groupby(["TS_TIPO_SERVICIO", "NOM_COL"])["DIF_FECHA"].mean().reset_index()
tmp_.columns = ['TIPO DE SERVICIO', "COLONIA", "DIAS EN ATENDER"]

tmp = tmp_.groupby(["TIPO DE SERVICIO"])["DIAS EN ATENDER"].mean()
fig_diff_time_tipo = px.bar(tmp.dropna().reset_index(), x="TIPO DE SERVICIO", y="DIAS EN ATENDER",
        text="DIAS EN ATENDER", color="DIAS EN ATENDER", color_continuous_scale="Aggrnyl")
fig_diff_time_tipo.update_traces(texttemplate='%{text:.2s}', textposition='auto')
fig_diff_time_tipo.update_layout(
    font_color=colors['title-dark'],
    paper_bgcolor=colors['background-figures-dark'],
    plot_bgcolor=colors['background-figures-dark'])
min_tipo =  tmp.nsmallest(1)
max_tipo =  tmp.nlargest(1)
tmp = tmp_.groupby(["COLONIA"])["DIAS EN ATENDER"].mean()
fig_diff_time_colonia = px.bar(tmp.dropna().reset_index(), x="COLONIA", y="DIAS EN ATENDER",
        text="DIAS EN ATENDER", color="DIAS EN ATENDER", color_continuous_scale="Aggrnyl")
fig_diff_time_colonia.update_traces(texttemplate='%{text:.2s}', textposition='auto')
fig_diff_time_colonia.update_layout(
    font_color=colors['title-dark'],
    paper_bgcolor=colors['background-figures-dark'],
    plot_bgcolor=colors['background-figures-dark'])
min_col =  tmp.nsmallest(1)
max_col =  tmp.nlargest(1)

nom = min_col.index[0]
days = min_col.values[0]
msg_col_min = f"#### La colonia donde más rápido se solucionan los problemas es **{nom}** ({round(days)} días)"
nom = max_col.index[0]
days = max_col.values[0]
msg_col_max = f"#### Y la colonia donde más tarda la atención es **{nom}** ({round(days)} días)"

nom = min_tipo.index[0]
days = min_tipo.values[0]
msg_tipo_min = f"#### Los reportes que más rápido se solucionan son **{nom}** ({round(days)} días)"
nom = max_tipo.index[0]
days = max_tipo.values[0]
msg_tipo_max = f"#### Y la reportes donde más tarda la atención es **{nom}** ({round(days)} días)"


## --------------------------- TIME IN CLOSING REPORTS----------------------

## --------------------------- REPORTS BY COLONIA----------------------

min_ = 10
pivot_1 = temp.groupby('COLONIA')['REPORTES'].count().reset_index()
others = {
    'COLONIA': 'OTRA', 
    'REPORTES': pivot_1.query(f"REPORTES < {min_}")['REPORTES'].sum()
}
pivot = pivot_1.query(f"REPORTES >= {min_}")
pivot = pivot.append(others, ignore_index=True)

fig_pie_reports_by_colonia = px.pie(pivot, 
             values='REPORTES', names='COLONIA', labels='COLONIA',
             color_discrete_sequence=px.colors.sequential.RdBu
        )
fig_pie_reports_by_colonia.update_traces(
    hovertemplate="""
        <b>%{label}</b><br>
        Reportes: %{value}
        """
)
fig_pie_reports_by_colonia.update_layout(
    font_color=colors['title-dark'],
    paper_bgcolor=colors['background-figures-dark'],
    plot_bgcolor=colors['background-figures-dark'])

max_col_r = pivot_1.nlargest(1, 'REPORTES')
min_col_r = pivot_1.nsmallest(1, 'REPORTES')
nom = min_col_r['COLONIA'].values[0]
reportes = min_col_r['REPORTES'].values[0]
msg_col_r_min = f"#### La colonia que menos reporta es **{nom}** ({round(reportes)} reportes)"
nom = max_col_r['COLONIA'].values[0]
reportes = max_col_r['REPORTES'].values[0]
msg_col_r_max = f"#### Y la que más lo hace es **{nom}** ({round(reportes)} reportes)"

## --------------------------- REPORTS BY COLONIA----------------------

## --------------------------- REPORTS BY DATE----------------------
pivot = df_zapopan.groupby(['SM_FECHA_REGISTRO', 'NOM_COL'])['SERV_MUNICIPAL_ID'].count().reset_index()
pivot.columns = ("FECHA", "COLONIA", "REPORTES")
fig_line_reports_by_date = px.line(pivot, x='FECHA', y='REPORTES', 
              color='COLONIA', labels='COLONIA', hover_name='COLONIA')
fig_line_reports_by_date.update_layout(    
    paper_bgcolor=colors['background-figures-ligth'],
    plot_bgcolor=colors['background-figures-ligth'],
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray'),
    )

## --------------------------- REPORTS BY DATE----------------------

## --------------------------- REPORTS BY TIPO----------------------
fig_status_by_tipo = px.sunburst(temp, path=["TIPO DE SERVICIO", "ESTATUS"], values="REPORTES",                                  
                  color_discrete_sequence=px.colors.sequential.Tealgrn,
                 color='ESTATUS', color_discrete_map=estatus_colors)
fig_status_by_tipo.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor=colors['background-figures-dark'],
    plot_bgcolor=colors['background-figures-dark'],

    )

pivot = temp.pivot_table(
    index='TIPO DE SERVICIO',
    columns='ESTATUS', 
    values='REPORTES', 
    aggfunc=sum)
pivot.fillna(0, inplace=True)
pivot['TOTAL'] = pivot['SIN ATENDER'] + pivot['EN PROCESO'] + pivot['ATENDIDO']
pivot['SIN ATENDER'] = pivot['SIN ATENDER'] + pivot['EN PROCESO']

max_tipo_total = pivot.nlargest(1, 'TOTAL')
min_tipo_total = pivot.nsmallest(1, 'TOTAL')
max_tipo_atendido = pivot.nlargest(1, 'ATENDIDO')
min_tipo_atendido = pivot.nlargest(1, 'SIN ATENDER')


msg_max_tipo_total = f"#### Los problemas que más se reportan son **{max_tipo_total.index[0]}**"
msg_min_tipo_total = f"#### Y los que son menos reportados son **{min_tipo_total.index[0]}**"

msg_max_tipo_atendido = f"#### Los reportes que más se atienden exitosamente son **{max_tipo_atendido.index[0]}**"
msg_min_tipo_atendido = f"#### Los reportes que menos se atienden son **{min_tipo_atendido.index[0]}**"
## --------------------------- REPORTS BY TIPO----------------------

## --------------------------- REPORTS BY COLONIA----------------------
fig_status_by_colonia = px.sunburst(temp, path=["COLONIA", "ESTATUS"], values="REPORTES",        
                  color_discrete_sequence=px.colors.sequential.Tealgrn,
                 color='ESTATUS', color_discrete_map=estatus_colors
                    )
fig_status_by_colonia.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor=colors['background-figures-dark'],
    plot_bgcolor=colors['background-figures-dark'],
    )
pivot = temp.pivot_table(
    index='COLONIA',
    columns='ESTATUS', 
    values='REPORTES', 
    aggfunc=sum)
pivot.fillna(0, inplace=True)
pivot['TOTAL'] = (pivot['SIN ATENDER'] + pivot['EN PROCESO'] + pivot['ATENDIDO'])
pivot['SIN ATENDER'] = pivot['SIN ATENDER'] + pivot['EN PROCESO']
pivot['PROPORCION'] = pivot['ATENDIDO']/pivot['TOTAL']
pivot
max_colonia_atendido = pivot.nlargest(1, 'ATENDIDO')
min_colonia_atendido = pivot.nlargest(1, 'SIN ATENDER')
max_colonia_atendido_prop = pivot.nlargest(1, 'PROPORCION')
min_colonia_atendido_prop = pivot.nsmallest(1, 'PROPORCION')

msg_max_colonia_atendido = f"#### La colonias que más se atenciones exitosas tiene es **{max_colonia_atendido.index[0]}** "
msg_min_colonia_atendido = f"#### Y la que menos tiene es **{min_colonia_atendido.index[0]}**"
msg_max_colonia_atendido_prop = f"#### La colonias que mayor proporción de atención tiene es **{max_colonia_atendido_prop.index[0]}** "
msg_min_colonia_atendido_prop = f"#### Y la que menos proporción de atendidos tiene es **{min_colonia_atendido_prop.index[0]}**"


## --------------------------- REPORTS BY COLONIA----------------------


header = lambda text: html.Div(html.H1(text), className="col-lg-12 col-md-12 col-12", 
    style={
    'textAlign': 'center', 
    'background':colors['background-figures-dark'],
    'margin-top': '10px',
    'padding': '10px',
    'color': colors['title-dark']
    })
seccion_header = lambda text: html.Div(html.H2(text), className="col-lg-12 col-md-12 col-12", style={'textAlign': 'center', 'margin-bottom': '5px'})
subseccion_header = lambda text: html.Div(html.H3(text), className="col-lg-12 col-md-12 col-12", style={'textAlign': 'center'})
subplot_header = lambda text: html.Div(html.H4(text), className="col-lg-12 col-md-12 col-12", style={'textAlign': 'center', 'color': colors['title-dark']})
markdown = lambda text: dcc.Markdown(text, style={'textAlign': 'center', 'padding': '10px'})

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=[
            {
                'name': 'viewport',
                'content': 'width=device-width, initial-scale=1.0'}])  # Init dashboard app

app.layout = html.Div([
    header('Resumen reportes de servicios municipales Zapopan'),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Markdown(f'## Hasta {today.strftime("%d-%m-%y")} ha habido **{total}** reportes', 
                    style={'textAlign': 'center', 'padding': '10px'}), sm=12, md=12, lg=12),
                dbc.Col(dbc.Alert(f'{atendidos} han sido atendidas y finalizadas', color='success'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(f'{en_proceso} están siendo atendidos', color='warning'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(f'{no_atendidos} no han sido atendidas aún', color='danger'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
            ], style=dict(padding='30px')),
            dbc.Row([
                dbc.Col(dcc.Markdown(f'## En el mes en curso ha habido **{total_this_month}** reportes', 
                    style={'textAlign': 'center', 'padding': '10px'}), sm=12, md=12, lg=12),
                dbc.Col(dbc.Alert(f'{atendidos_this_month} han sido atendidas y finalizadas', color='success'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(f'{en_proceso_this_month} están siendo atendidos', color='warning'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(f'{no_atendidos_this_month} no han sido atendidas aún', color='danger'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
            ], style=dict(padding='30px')),
            dbc.Row([
                dbc.Col(dcc.Markdown(f'## Hoy ha habido **{total_today_}** reportes', 
                    style={'textAlign': 'center', 'padding': '10px'}), sm=12, md=12, lg=12),
                dbc.Col(dbc.Alert(f'{atendidos_today_} han sido atendidas y finalizadas', color='success'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(f'{en_proceso_today_} están siendo atendidos', color='warning'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(f'{no_atendidos_today_} no han sido atendidas aún', color='danger'), sm=12, md=4, lg=4, style={'textAlign': 'center'}),
            ], style=dict(padding='30px')),
            ], sm=12, md=5, lg=6),
            dbc.Col([
                subplot_header("Todos los reportes"),
                dcc.Graph(id="fig-pie-estatus-total--", figure=fig_pie_estatus_total.update_layout(
                    legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center", x=0.5)))    
            ],sm=12, md=7, lg=6),   
        ], align='center'),

    # Colonias
    dbc.Row([
            dbc.Col([
                dbc.Alert([
                subplot_header("Reportes a lo largo del tiempo"),
                dcc.Graph(id="fig-line-reports-by-date", figure=fig_line_reports_by_date)], color='ligth')            
                ], sm=12, md=12, lg=12, align='center')
        ], align='center'),
    dbc.Row(seccion_header('Reportes por zonas')),
    dbc.Row([
        dbc.Col(
            dbc.Col([
                dbc.Alert([
                subplot_header("Número de reportes de acuerdo a la colonia"),
                dcc.Graph(id="fig-pie-reports-by-colonia", figure=fig_pie_reports_by_colonia)], color='dark')            
                ]), sm=12, md=7, lg=8, align='center'
            ),
        dbc.Col([
                dbc.Col(dbc.Alert(markdown(msg_col_r_min), color='ligth'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_col_r_max), color='ligth'), style={'textAlign': 'center'}),
                # dbc.Col(dbc.Alert(, color='danger'), style={'textAlign': 'center'}),
                ], sm=12, md=5, lg=4), 

        ], align='center'),
    dbc.Row([
        dbc.Col([
                dbc.Col(dbc.Alert(markdown(msg_max_colonia_atendido), color='dark'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_min_colonia_atendido), color='ligth'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_max_colonia_atendido_prop), color='dark'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_min_colonia_atendido_prop), color='ligth'), style={'textAlign': 'center'}),
                # dbc.Col(dbc.Alert(, color='danger'), style={'textAlign': 'center'}),
                ], sm=12, md=5, lg=4), 
        dbc.Col(
            dbc.Col([
                dbc.Alert([
                subplot_header("Atención a reportes de acuerdo a la colonia"),
                dcc.Graph(id="fig'status'by'colonia", figure=fig_status_by_colonia)], color='dark')            
                ]), sm=12, md=7, lg=8, align='center'
            ),

        ], align='center'),
    # dbc.Row(
    #     dbc.Col([
    #         dbc.Alert([
    #             subseccion_header("Atención a reportes de acuerdo a la colonia"),
    #             dcc.Graph(id="fig-bar-status-by-tipo", figure=fig_bar_status_by_tipo)], color='dark')            
    #     ],sm=12, md=12, lg=12, style=dict(padding='30px')),
    # ),

    dbc.Row([
        seccion_header("Estatus de reportes:"),
        dbc.Col([
            subplot_header("Totales"),
            dcc.Graph(id="fig-pie-estatus-total", figure=fig_pie_estatus_total)    
        ],sm=12, md=12, lg=4),   
        dbc.Col([
            subplot_header("Este mes"),
            dcc.Graph(id="fig-pie-estatus-this-month", figure=fig_pie_estatus_this_month)    
        ],sm=12, md=12, lg=4),   
        dbc.Col([
            subplot_header("La semana en curso"),
            dcc.Graph(id="fig-pie-estatus-this-week", figure=fig_pie_estatus_this_week)    
        ],sm=12, md=12, lg=4),
    ]),
    dbc.Row([
        dbc.Col(dcc.Markdown(msg_diff_cierre, style={'textAlign': 'center', 'padding': '10px'}), sm=12, md=12, lg=12)], style=dict(padding='30px')),
    dbc.Row([
        dbc.Col([
                dbc.Col(dbc.Alert(markdown(msg_col_min), color='ligth'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_col_max), color='ligth'), style={'textAlign': 'center'}),
                # dbc.Col(dbc.Alert(, color='danger'), style={'textAlign': 'center'}),
                ], sm=12, md=5, lg=4), 
        dbc.Col(
            dbc.Col([
                dbc.Alert([
                    subplot_header("Tiempo de atención por colonia"),
                    dcc.Graph(id="fig-diff-time-colonia", figure=fig_diff_time_colonia)], color='dark')                
                ]), sm=12, md=7, lg=8, align='center'
            ),

        ], align='center'),
    dbc.Row([
        dbc.Col(
            dbc.Col([
                dbc.Alert([
                    subplot_header("Tiempo de atención por tipo de reportes"),
                    dcc.Graph(id="fig-diff-time-tipo", figure=fig_diff_time_tipo)], color='dark')                
                ]), sm=12, md=7, lg=8, align='center'
            ),
        dbc.Col([
                dbc.Col(dbc.Alert(markdown(msg_tipo_min), color='ligth'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_tipo_max), color='ligth'), style={'textAlign': 'center'}),
                # dbc.Col(dbc.Alert(, color='danger'), style={'textAlign': 'center'}),
                ], sm=12, md=5, lg=4), 

        ], align='center'),
    seccion_header("Reportes de acuerdo al tipo de problema"),
    dbc.Row([
        dbc.Col([
                dbc.Col(dbc.Alert(markdown(msg_max_tipo_total), color='dark'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_min_tipo_total), color='ligth'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_max_tipo_atendido), color='dark'), style={'textAlign': 'center'}),
                dbc.Col(dbc.Alert(markdown(msg_min_tipo_atendido), color='ligth'), style={'textAlign': 'center'}),
                # dbc.Col(dbc.Alert(, color='danger'), style={'textAlign': 'center'}),
                ], sm=12, md=5, lg=4), 
        dbc.Col(
            dbc.Col([
                dbc.Alert([
                    subplot_header("Atención a los reportes de acuerdo al tipo de problema"),
                    dcc.Graph(id="fig-status-by-tipo", figure=fig_status_by_tipo)], color='dark')                
                ]), sm=12, md=7, lg=8, align='center'
            ),

        ], align='center'),

], style={'overflow-y':'100%'})
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False, port=5002)