import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
from streamlit_plotly_events import plotly_events
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Análisis de Energía",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_data():
    """Carga los datos del CSV y los preprocesa para análisis"""
    data = pd.read_csv("World Energy Consumption.csv")
    data['year'] = pd.to_datetime(data['year'], format='%Y')
    return data

data = load_data()

ELECTRICITY_COLUMNS = {
    'generacion': [
        'low_carbon_electricity', 'nuclear_electricity', 'oil_electricity',
        'other_renewable_electricity', 'other_renewable_exc_biofuel_electricity',
        'solar_electricity', 'wind_electricity', 'fossil_electricity',
        'gas_electricity', 'hydro_electricity', 'biofuel_electricity',
        'coal_electricity'
    ],
    'consumo': [
        'biofuel_consumption', 'coal_consumption', 'fossil_fuel_consumption',
        'gas_consumption', 'hydro_consumption', 'low_carbon_consumption',
        'nuclear_consumption', 'oil_consumption', 'solar_consumption'
    ]
}

CONTINENTS = {
    'África': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde',
               'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Cote d\'Ivoire',
               'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea',
               'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya',
               'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania',
               'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda',
               'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia',
               'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia',
               'Uganda', 'Zambia', 'Zimbabwe'],
    'Asia': ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan',
             'Brunei Darussalam', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India',
             'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait',
             'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia','Maldives', 'Mongolia', 'Myanmar', 'Nepal',
             'North Korea', 'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia',
             'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan',
             'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates',
             'Uzbekistan', 'Vietnam', 'Yemen'],
    'Europa': ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
               'Bulgaria', 'Croatia', 'Czech Republic', 'Denmark', 'Estonia', 'Finland',
               'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
               'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta',
               'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway',
               'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia',
               'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City'],
    'Norte América': ['Canada', 'Greenland', 'Mexico', 'United States'],
    'Oceanía': ['Australia', 'Fiji', 'New Zealand', 'Papua New Guinea'],
    'Sud América': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',
                  'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'],
}

PROPHET_PARAMS = {
    'United States': {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0,
                     'seasonality_mode': 'multiplicative', 'train_size': 0.7,
                     'best_rmse': 174.61},
    'China': {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 10.0,
             'seasonality_mode': 'multiplicative', 'train_size': 0.7,
             'best_rmse': 2838.10},
    'Japan': {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 1.0,
             'seasonality_mode': 'multiplicative', 'train_size': 0.7,
             'best_rmse': 76.16},
    'Russia': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.01,
              'seasonality_mode': 'multiplicative', 'train_size': 0.7,
              'best_rmse': 31.73},
    'India': {'changepoint_prior_scale': 1, 'seasonality_prior_scale': 0.1,
             'seasonality_mode': 'additive', 'train_size': 0.7,
             'best_rmse': 429.25},
    'Germany': {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0,
               'seasonality_mode': 'multiplicative', 'train_size': 0.7,
               'best_rmse': 81.73},
    'Canada': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0,
              'seasonality_mode': 'additive', 'train_size': 0.7,
              'best_rmse': 47.45},
    'France': {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 1.0,
              'seasonality_mode': 'multiplicative', 'train_size': 0.7,
              'best_rmse': 76.11},
    'Brazil': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.1,
              'seasonality_mode': 'multiplicative', 'train_size': 0.7,
              'best_rmse': 95.31},
    'United Kingdom': {'changepoint_prior_scale': 1, 'seasonality_prior_scale': 10.0,
                      'seasonality_mode': 'multiplicative', 'train_size': 0.7,
                      'best_rmse': 40.54},
    'South Korea': {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.01,
                   'seasonality_mode': 'multiplicative', 'train_size': 0.7,
                   'best_rmse': 27.14},
    'Italy': {'changepoint_prior_scale': 1, 'seasonality_prior_scale': 10.0,
             'seasonality_mode': 'additive', 'train_size': 0.7,
             'best_rmse': 59.25},
    'Spain': {'changepoint_prior_scale': 1, 'seasonality_prior_scale': 10.0,
             'seasonality_mode': 'multiplicative', 'train_size': 0.7,
             'best_rmse': 83.69},
    'Mexico': {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.1,
              'seasonality_mode': 'multiplicative', 'train_size': 0.7,
              'best_rmse': 12.39},
    'South Africa': {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.01,
                    'seasonality_mode': 'multiplicative', 'train_size': 0.7,
                    'best_rmse': 28.99},

    # Continentes
    'África': {'train_size': 0.7, 'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1,
              'seasonality_mode': 'multiplicative'},
    'Asia': {'train_size': 0.7, 'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1,
            'seasonality_mode': 'multiplicative'},
    'Europa': {'train_size': 0.7, 'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1,
              'seasonality_mode': 'additive'},
    'Norte América': {'train_size': 0.7, 'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 0.1,
                     'seasonality_mode': 'additive'},
    'Oceanía': {'train_size': 0.7, 'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0,
               'seasonality_mode': 'additive'},
    'Sud América': {'train_size': 0.6, 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.01,
                   'seasonality_mode': 'additive'},
    'Mundial': {'train_size': 0.7, 'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0,
               'seasonality_mode': 'additive'},
}

RENAME_COLUMNS = {
    'nuclear_electricity': 'nuclear',
    'oil_electricity': 'petroleo',
    'solar_electricity': 'solar',
    'wind_electricity': 'eolica',
    'fossil_electricity': 'fosil',
    'gas_electricity': 'gas',
    'hydro_electricity': 'hidro',
    'biofuel_electricity': 'bio fuel',
}

def mapa_continente():
    """Construye mapa completo de continentes y países"""
    continent_map = CONTINENTS.copy()
    for country in PROPHET_PARAMS.keys():
        if country not in continent_map:
            continent_map[country] = [country]

    continent_map['Mundial'] = []
    for continent, countries in continent_map.items():
        if continent != 'Mundial':
            continent_map['Mundial'].extend(countries)

    return continent_map

continent_map = mapa_continente()

def data_energia():
    """Prepara datos de generación y consumo de electricidad por país/continente"""
    gen_data = data.groupby(['year','country'])[ELECTRICITY_COLUMNS['generacion']].sum().sum(axis=1).to_frame('electricity_generation').reset_index("country")

    cons_data = data.groupby(["year", "country"])[ELECTRICITY_COLUMNS['consumo']].sum().sum(axis=1).to_frame('electricity_consumption').reset_index('country')

    combined_data = pd.merge(gen_data, cons_data, on=['year', 'country'])

    combined_data['continent'] = combined_data['country'].map(
        lambda x: next((k for k, v in continent_map.items() if x in v), None)
    )

    filtered_data = combined_data[
        (combined_data['continent'].notna()) &
        (combined_data.index >= pd.Timestamp('1985-01-01'))
    ]

    return filtered_data

def data_continente():
    """Prepara datos agregados por continente"""
    country_data = data_energia()

    continent_data = country_data.groupby(['year', 'continent'])[
        ['electricity_generation', 'electricity_consumption']
    ].sum().reset_index('continent')

    continents_data = {}
    for continent in continent_map.keys():
        continent_subset = continent_data[continent_data['continent'] == continent].copy()

        if continent == 'África' and not continent_subset.empty:
            continent_subset = continent_subset[continent_subset.index < pd.Timestamp('2022-01-01')]

        if not continent_subset.empty:
            continents_data[continent] = continent_subset

    mundial_data = continent_data.groupby('year')[
        ['electricity_generation', 'electricity_consumption']
    ].sum()
    mundial_data['continent'] = 'Mundial'
    continents_data['Mundial'] = mundial_data

    return continents_data

def data_prophet():
    """Prepara datos para Prophet con variables exógenas"""
    prophet_data = {}

    continent_data = data_continente()
    for continent, df in continent_data.items():
        if continent in PROPHET_PARAMS:
            prophet_df = df.reset_index()
            prophet_df['ds'] = pd.to_datetime(prophet_df['year'])
            prophet_df['y'] = prophet_df['electricity_generation']
            prophet_data[continent] = (prophet_df, PROPHET_PARAMS[continent]['train_size'])

    country_data = data_energia()
    for country in PROPHET_PARAMS.keys():
        if country not in prophet_data:
            country_subset = country_data[country_data['country'] == country].copy()

            if not country_subset.empty:
                prophet_df = country_subset.reset_index()
                prophet_df['ds'] = pd.to_datetime(prophet_df['year'])
                prophet_df['y'] = prophet_df['electricity_generation']
                prophet_data[country] = (prophet_df, PROPHET_PARAMS[country]['train_size'])

    return prophet_data


def plot_energy_distribution(region, data):
    """Visualiza la distribución de fuentes de energía para una región"""
    is_continent = region in ['Mundial', 'África', 'Asia', 'Europa', 'Norte América', 'Oceanía', 'Sud América']

    if region == 'Mundial':
        energy_data = data.groupby(['year'])[ELECTRICITY_COLUMNS['generacion']].sum()
    elif is_continent:
        data_copy = data.copy()
        data_copy['continente'] = data_copy['country'].map(
            lambda x: next((k for k, v in continent_map.items() if x in v), None)
        )
        energy_data = data_copy[data_copy['continente'] == region].groupby(['year'])[ELECTRICITY_COLUMNS['generacion']].sum()
    else:
        energy_data = data[data['country'] == region].groupby(['year'])[ELECTRICITY_COLUMNS['generacion']].sum()

    if not isinstance(energy_data.index, pd.DatetimeIndex):
        energy_data.index = pd.to_datetime(energy_data.index)

    energy_data = energy_data[energy_data.index.year >= 1985]

    energy_data['carbon'] = energy_data['low_carbon_electricity'] + energy_data['coal_electricity']
    energy_data['renovable'] = energy_data['other_renewable_electricity'] + energy_data['other_renewable_exc_biofuel_electricity']

    energy_data = energy_data.drop(columns=[
        'low_carbon_electricity', 'other_renewable_electricity',
        'other_renewable_exc_biofuel_electricity', 'coal_electricity'
    ])

    energy_sources = ['nuclear', 'petroleo', 'solar', 'eolica', 'fosil',
                     'gas', 'hidro', 'bio fuel', 'carbon', 'renovable']

    energy_data = energy_data.rename(columns=RENAME_COLUMNS)

    # Crear dos figuras separadas
    fig1 = go.Figure()

    # Agregar trazas para la evolución de la producción (línea)
    for source in energy_sources:
        fig1.add_trace(
            go.Scatter(
                x=energy_data.index.year,
                y=energy_data[source],
                mode='lines+markers',
                name=source,
                marker=dict(size=6)
            )
        )

    fig1.update_layout(
        title=f'Evolución de la producción de energía en {region}',
        xaxis_title="Año",
        yaxis_title="TWh",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Posicionar leyenda a la derecha del gráfico
            font=dict(size=10)
        ),
        width=500,
        height=600,
        margin=dict(t=50, b=50, l=50, r=100)  # Aumentar margen derecho para la leyenda
    )

    # Crear figura para el gráfico de pastel
    fig2 = go.Figure()

    # Calcular totales y porcentajes para el gráfico de pastel
    totals = energy_data[energy_sources].sum()
    porcentajes = (totals / totals.sum()) * 100

    # Agregar la traza de pastel (donut)
    fig2.add_trace(
        go.Pie(
            labels=totals.index,
            values=porcentajes,
            hole=0.5,
            textinfo='percent',
            textposition='inside',
            insidetextorientation='radial',
            textfont=dict(size=10),
            marker=dict(line=dict(color='#000000', width=1))
        )
    )

    fig2.update_layout(
        title=f'Distribución Energía {region}',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Posicionar leyenda a la derecha del gráfico
            font=dict(size=10)
        ),
        width=500,
        height=600,
        margin=dict(t=50, b=50, l=50, r=100)  # Aumentar margen derecho para la leyenda
    )

    # Crear un subplot para combinar ambas figuras
    subfig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        horizontal_spacing=0.05  # Reducir el espacio entre los gráficos
    )

    # Transferir las trazas de fig1 a subfig
    for trace in fig1.data:
        subfig.add_trace(trace, row=1, col=1)

    # Transferir las trazas de fig2 a subfig
    for trace in fig2.data:
        subfig.add_trace(trace, row=1, col=2)

    # Actualizar el layout final
    subfig.update_layout(
        width=1000,
        height=600,
        showlegend=True
    )

    # Aplicar títulos
    subfig.update_xaxes(title_text="Año", row=1, col=1)
    subfig.update_yaxes(title_text="TWh", row=1, col=1, rangemode="tozero")

    # CORRECCIÓN: Eliminar agrupación de leyendas para permitir toggle individual
    for i in range(len(subfig.data)):
        if i < len(fig1.data):  # Para las trazas del gráfico de líneas
            subfig.data[i].showlegend = True

        else:  # Para las trazas del gráfico de pastel
            subfig.data[i].showlegend = True
            subfig.data[i].legendgroup = "group2"
            subfig.data[i].legendgrouptitle = dict(text="Distribución")

    return subfig



def prediction_prophet(region, target_year, prophet_data_dict):
    """Genera predicciones usando Prophet con variable exógena de consumo"""

    prophet_df, train_size = prophet_data_dict[region]
    params = PROPHET_PARAMS[region]

    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        seasonality_mode=params['seasonality_mode']
    )

    model.add_regressor('electricity_consumption')
    model.fit(prophet_df)

    last_date = prophet_df['ds'].max()
    years_to_predict = target_year - last_date.year

    future = model.make_future_dataframe(periods=years_to_predict, freq='Y')
    future = future.merge(
        prophet_df[['ds', 'electricity_consumption']],
        on='ds',
        how='left'
    )

    # Completar valores faltantes usando forward fill (último valor conocido)
    future['electricity_consumption'] = future['electricity_consumption'].fillna(method='ffill')

    forecast = model.predict(future)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='lines+markers',
        name='Datos reales',
        line=dict(color='red', width=3)
    ))

    future_forecast = forecast[forecast['ds'] > last_date]

    if not future_forecast.empty:
        # Añadir línea que conecta los datos reales con la predicción
        fig.add_trace(go.Scatter(
            x=[prophet_df['ds'].iloc[-1], future_forecast['ds'].iloc[0]],
            y=[prophet_df['y'].iloc[-1], future_forecast['yhat'].iloc[0]],
            mode='lines+markers',
            line=dict(color='red', width=3),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='Predicción (con consumo)',
            line=dict(color='blue', width=3)
        ))

    fig.update_layout(
        title=f"Predicción de Generación de Electricidad - {region}",
        xaxis_title="Año",
        yaxis_title="Generación de electricidad (TWh)",
        xaxis=dict(range=[prophet_df['ds'].min(), pd.Timestamp(f'{target_year}-12-31')]),
        legend=dict(x=0, y=1),
        height=800
    )

    forecast_table = forecast[['ds', 'yhat']].copy()
    forecast_table['ds'] = forecast_table['ds'].dt.year
    forecast_table['ds'] = forecast_table['ds'] + 1
    forecast_table = forecast_table[
        (forecast_table['ds'] >= 2023) &
        (forecast_table['ds'] <= target_year)
    ]

    forecast_table['ds'] = forecast_table['ds'].astype(str)
    forecast_table['yhat'] = forecast_table['yhat'].round(2).astype(str).str.replace('.', ',')
    forecast_table.columns = ['Año', 'Predicción - TWh']

    return fig, forecast_table

    # if not future_forecast.empty:
    #     ax.plot(
    #         [prophet_df['ds'].iloc[-1], future_forecast['ds'].iloc[0]],
    #         [prophet_df['y'].iloc[-1], future_forecast['yhat'].iloc[0]],
    #         'r-',
    #         linewidth=1.5
    #     )

    #     ax.plot(
    #         future_forecast['ds'],
    #         future_forecast['yhat'],
    #         'b-',
    #         linewidth=1.5,
    #         label='Predicción (con consumo)'
    #     )

        # ax.fill_between(
        #     future_forecast['ds'],
        #     future_forecast['yhat_lower'],
        #     future_forecast['yhat_upper'],
        #     color='blue',
        #     alpha=0.1,
        #     label='Intervalo de confianza'
        # )

    # ax.set_title(f"Predicción de Generación de Electricidad - {region}", pad=20)
    # ax.set_xlabel("Año")
    # ax.set_ylabel("Generación de electricidad (TWh)")
    # ax.legend(loc='upper left')
    # ax.grid(True, linestyle='-', alpha=0.2)
    # ax.set_xlim(prophet_df['ds'].min(), pd.Timestamp(f'{target_year}-12-31'))

def mapa_region(selected_region):
    """Crea un mapa interactivo para la región seleccionada"""
    iso_codes = {
        'África': ['DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CPV', 'CMR', 'CAF',
                  'TCD', 'COM', 'COG', 'CIV', 'COD', 'DJI', 'EGY', 'GNQ', 'ERI',
                  'SWZ', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB', 'KEN', 'LSO',
                  'LBR', 'LBY', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MAR', 'MOZ',
                  'NAM', 'NER', 'NGA', 'RWA', 'STP', 'SEN', 'SYC', 'SLE', 'SOM',
                  'ZAF', 'SSD', 'SDN', 'TZA', 'TGO', 'TUN', 'UGA', 'ZMB', 'ZWE'],
        'Asia': ['AFG', 'ARM', 'AZE', 'BHR', 'BGD', 'BTN', 'BRN', 'KHM', 'CHN',
                'CYP', 'GEO', 'IND', 'IDN', 'IRN', 'IRQ', 'ISR', 'JPN', 'JOR',
                'KAZ', 'KWT', 'KGZ', 'LAO', 'LBN', 'MYS', 'MDV', 'MNG', 'MMR',
                'NPL', 'PRK', 'OMN', 'PAK', 'PSE', 'PHL', 'QAT', 'SAU', 'SGP',
                'KOR', 'LKA', 'SYR', 'TWN', 'TJK', 'THA', 'TLS', 'TUR', 'TKM',
                'ARE', 'UZB', 'VNM', 'YEM'],
        'Europa': ['ALB', 'AND', 'AUT', 'BLR', 'BEL', 'BIH', 'BGR', 'HRV', 'CZE',
                  'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'ISL', 'IRL',
                  'ITA', 'XKX', 'LVA', 'LIE', 'LTU', 'LUX', 'MLT', 'MDA', 'MCO',
                  'MNE', 'NLD', 'MKD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SMR',
                  'SRB', 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'UKR', 'GBR', 'VAT'],
        'Norte América': ['CAN', 'GRL', 'MEX', 'USA'],
        'Oceanía': ['AUS', 'FJI', 'NZL', 'PNG'],
        'Sud América': ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'GUY', 'PRY',
                       'PER', 'SUR', 'URY', 'VEN'],

        'United States': ['USA'],
        'China': ['CHN'],
        'Japan': ['JPN'],
        'Russia': ['RUS'],
        'India': ['IND'],
        'Germany': ['DEU'],
        'Canada': ['CAN'],
        'France': ['FRA'],
        'Brazil': ['BRA'],
        'United Kingdom': ['GBR'],
        'South Korea': ['KOR'],
        'Italy': ['ITA'],
        'Spain': ['ESP'],
        'Mexico': ['MEX'],
        'South Africa': ['ZAF']
    }

    fig = go.Figure()

    if selected_region == 'Mundial':
        all_countries = []
        for codes in iso_codes.values():
            all_countries.extend(codes)

        fig.add_trace(go.Choropleth(
            locations=all_countries,
            z=[1] * len(all_countries),
            colorscale=[[0, 'rgb(200, 200, 255)'], [1, 'rgb(200, 200, 255)']],
            showscale=False,
            marker_line_color='rgb(100, 100, 100)',
            marker_line_width=0.5
        ))
    else:
        countries = iso_codes.get(selected_region, [])
        fig.add_trace(go.Choropleth(
            locations=countries,
            z=[1] * len(countries),
            colorscale=[[0, 'rgb(100, 149, 237)'], [1, 'rgb(100, 149, 237)']],
            showscale=False,
            marker_line_color='rgb(50, 50, 50)',
            marker_line_width=0.5
        ))

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="black",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="lightblue",
        projection_type="equirectangular",
        showframe=False,
        showcountries=True,
        countrycolor="gray",
        showlakes=True,
        lakecolor="lightblue"
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        geo=dict(
            showframe=False,
            showcoastlines=True,
        )
    )

    return fig

def main():
    st.title("Análisis y predicción de energía por continente")
    st.sidebar.header("Configuración")

    # Cargar datos para Prophet
    prophet_data = data_prophet()

    # Controles de la aplicación
    control_col1, control_col2, control_col3 = st.columns([1, 1, 2])

    with control_col1:
        available_regions = list(PROPHET_PARAMS.keys())
        selected_region = st.selectbox(
            "Selecciona una región o país",
            available_regions,
            index=available_regions.index("Mundial") if "Mundial" in available_regions else 0
        )

    with control_col2:
        prediction_year = st.slider(
            'Seleccionar año de predicción:',
            min_value=2024,
            max_value=2035,
            value=2030,
            step=1
        )

    # Generar predicciones
    if selected_region in prophet_data:
        fig_forecast, forecast_data = prediction_prophet(
            selected_region, prediction_year, prophet_data
        )
    else:
        st.warning(f"No hay datos disponibles para {selected_region}")
        fig_forecast = plt.figure()
        forecast_data = pd.DataFrame(columns=['Año', 'Predicción - TWh'])

    # Mostrar predicciones
    st.subheader("Predicción de Generación Eléctrica")
    col_tabla, col_prediccion = st.columns([1, 2])

    with col_tabla:
        st.markdown("### Valores Predichos y Métricas")
        st.dataframe(
            forecast_data[1:],
            hide_index=True,
            use_container_width=True
        )

        # Mostrar mapa de la región
        fig_map = mapa_region(selected_region)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_prediccion:
        st.plotly_chart(fig_forecast, use_container_width=True)

    # Mostrar distribución de energía
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Distribución de Energía</h2>",
               unsafe_allow_html=True)

    col_central = st.columns([1])[0]
    with col_central:
        fig_distribution = plot_energy_distribution(selected_region, data)
        st.plotly_chart(fig_distribution, use_container_width=True)

if __name__ == "__main__":
    main()
