import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Análisis de Energía",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

@st.cache_data
def load_data():
    """Carga los datos del CSV y los preprocesa para análisis"""
    data = pd.read_csv("World Energy Consumption.csv")
    data['year'] = pd.to_datetime(data['year'], format='%Y')
    valid_countries = [country for countries in CONTINENTS.values() for country in countries]

    # Filtramos el DataFrame conservando solo las filas cuyo valor en 'country' esté en valid_countries
    data = data[data['country'].isin(valid_countries)]

    # Si quieres resetear el índice, puedes hacerlo así:
    data = data.reset_index(drop=True)

    return data

data = load_data()

ELECTRICITY_COLUMNS = {
    'generacion': [
        'nuclear_electricity', 'oil_electricity',
        'solar_electricity', 'wind_electricity',
        'gas_electricity', 'hydro_electricity',
        'biofuel_electricity','coal_electricity'

    ],
    'consumo': [
        'nuclear_consumption', 'oil_consumption',
        'solar_consumption','wind_consumption',
        'gas_consumption', 'hydro_consumption',
        'biofuel_consumption', 'coal_consumption',
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
    'coal_electricity': 'carbon',
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

    print(data)
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

    # energy_data['carbon'] = energy_data['low_carbon_electricity'] + energy_data['coal_electricity']
    # energy_data['renovable'] = energy_data['other_renewable_electricity'] + energy_data['other_renewable_exc_biofuel_electricity']

    # energy_data = energy_data.drop(columns=[
    #     'low_carbon_electricity', 'other_renewable_electricity',
    #     'other_renewable_exc_biofuel_electricity', 'coal_electricity'
    # ])
    energy_sources = ['nuclear', 'petroleo', 'solar', 'eolica',
                     'gas', 'hidro', 'bio fuel', 'carbon']

    energy_data = energy_data.rename(columns=RENAME_COLUMNS)



    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Evolución de la producción de energía en {region}',
                        f'Distribución Energía {region}'),
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    # Agregar trazas para la evolución de la producción (línea)
    for source in energy_sources:
        fig.add_trace(
            go.Scatter(
                x=energy_data.index.year,
                y=energy_data[source],
                mode='lines+markers',
                name=source
            ),
            row=1, col=1
        )
    fig.update_layout(xaxis=dict(showgrid=True, griddash='dot'))
    fig.update_layout(yaxis=dict(showgrid=True, griddash='dot'))

    # Calcular totales y porcentajes para el gráfico de pastel
    totals = energy_data[energy_sources].sum()
    porcentajes = (totals / totals.sum()) * 100

    # Reordenar los totales y porcentajes según el orden de 'energy_sources'
    totals = totals[energy_sources]
    porcentajes = porcentajes[energy_sources]

    # Agregar la traza de pastel (donut)
    # Agregar la traza de pastel (donut) con el mismo orden
    fig.add_trace(
        go.Pie(
            #labels=totals.index,  # Utilizamos el índice reordenado
            labels=energy_sources,
            values=porcentajes,
            hole=0.5,
            textinfo='percent+label',
            legendgroup='energy_sources', # Sincronizar leyenda con el grupo
            showlegend=True
        ),
        row=1, col=2
    )

    # Ajustar el layout
    #fig.update_layout(margin=dict(t=50, b=0, l=0, r=0))
    fig.update_layout(width=1400,height=550,margin=dict(t=50, b=50, l=0, r=50),legend_title="Gráfico de Líneas")

    return fig



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
        line=dict(color='red', width=2)
    ))


    future_forecast = forecast[forecast['ds'] > last_date]

    if not future_forecast.empty:
        # Añadir línea que conecta los datos reales con la predicción
        fig.add_trace(go.Scatter(
            x=[prophet_df['ds'].iloc[-1], future_forecast['ds'].iloc[0]],
            y=[prophet_df['y'].iloc[-1], future_forecast['yhat'].iloc[0]],
            mode='lines+markers',
            line=dict(color='red', width=2),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='Predicción (con consumo)',
            line=dict(color='blue', width=3)
        ))
    fig.update_layout(xaxis=dict(showgrid=True, griddash='dot'))
    fig.update_layout(yaxis=dict(showgrid=True, griddash='dot'))

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

def prophet_energias(data, target_year=2030):

    col_electricity = [
        'low_carbon_electricity', 'nuclear_electricity', 'oil_electricity',
        'other_renewable_electricity', 'other_renewable_exc_biofuel_electricity',
        'solar_electricity', 'wind_electricity', 'gas_electricity',
        'hydro_electricity', 'biofuel_electricity', 'coal_electricity'
    ]

    col_cons = [
        'nuclear_consumption', 'oil_consumption', 'solar_consumption',
        'wind_consumption', 'fossil_fuel_consumption', 'gas_consumption',
        'hydro_consumption', 'biofuel_consumption', 'coal_consumption',
        'low_carbon_consumption'
    ]

    rename_columns = {
        'nuclear_electricity': 'nuclear',
        'oil_electricity': 'petroleo',
        'solar_electricity': 'solar',
        'wind_electricity': 'eolica',
        'fossil_electricity': 'fosil',
        'gas_electricity': 'gas',
        'hydro_electricity': 'hidro',
        'biofuel_electricity': 'bio fuel',
    }

    rename_columns_cons = {
        'nuclear_consumption': 'nuclear',
        'oil_consumption': 'petroleo',
        'solar_consumption': 'solar',
        'wind_consumption': 'eolica',
        'fossil_fuel_consumption': 'fosil',
        'gas_consumption': 'gas',
        'hydro_consumption': 'hidro',
        'biofuel_consumption': 'bio fuel',
    }

    columns = [
        'nuclear', 'petroleo', 'solar', 'eolica', 'gas',
        'hidro', 'bio fuel', 'carbon'
    ]


    color_map = {
        'nuclear': {'historical': '#6A1B9A', 'prediction': '#9C27B0'},  # Púrpura oscuro → Púrpura claro
        'petroleo': {'historical': '#D32F2F', 'prediction': '#FF7043'},  # Rojo fuerte → Naranja rojizo
        'solar': {'historical': '#FFA000', 'prediction': '#FFEB3B'},  # Naranja oscuro → Amarillo brillante
        'eolica': {'historical': '#1B5E20', 'prediction': '#4CAF50'},  # Verde oscuro → Verde brillante
        'gas': {'historical': '#1565C0', 'prediction': '#42A5F5'},  # Azul fuerte → Azul más claro
        'hidro': {'historical': '#0277BD', 'prediction': '#81D4FA'},  # Azul profundo → Celeste
        'bio fuel': {'historical': '#4E342E', 'prediction': '#8D6E63'},  # Marrón oscuro → Marrón claro
        'carbon': {'historical': '#424242', 'prediction': '#BDBDBD'}  # Gris oscuro → Gris claro
    }

    fig_mundial = go.Figure()
    fig_energia = go.Figure()

    # Asegurarse de que año es un valor entero, no datetime
    data_copy = data.copy()
    if pd.api.types.is_datetime64_any_dtype(data_copy['year']):
        data_copy['year'] = data_copy['year'].dt.year

    mundial = data_copy.groupby('year')[col_electricity].sum().reset_index()
    mundial['y'] = mundial[col_electricity].sum(axis=1)
    mundial.drop(columns=col_electricity, inplace=True)
    mundial['ds'] = mundial['year']
    mundial.drop(columns=['year'], inplace=True)
    mundial = mundial.groupby('ds').sum().reset_index()

    # Generación datos de producción ordenados
    produccion = data_copy.groupby('year')[col_electricity].sum().reset_index()
    produccion = produccion[produccion['year'] > 1985].reset_index(drop=True)
    produccion['carbon'] = produccion['low_carbon_electricity'] + produccion['coal_electricity']
    produccion['renovable'] = produccion['other_renewable_electricity'] + produccion['other_renewable_exc_biofuel_electricity']
    produccion.drop(columns=['low_carbon_electricity', 'other_renewable_electricity',
                        'other_renewable_exc_biofuel_electricity', 'coal_electricity'], inplace=True)
    produccion.rename(columns=rename_columns, inplace=True)

    # Generación datos de consumo ordenados
    consumo = data_copy.groupby('year')[col_cons].sum().reset_index()
    consumo = consumo[consumo['year'] > 1985].reset_index(drop=True)
    consumo['carbon'] = consumo['low_carbon_consumption'] + consumo['coal_consumption']
    consumo.drop(columns=['low_carbon_consumption', 'coal_consumption'], inplace=True)
    consumo.rename(columns=rename_columns_cons, inplace=True)

    # Modelo predictivo mundial y gráfico
    p = data_copy.groupby('year')[col_cons].sum().sum(axis=1).to_frame('consumo').reset_index()
    mundial = mundial.merge(p, left_on='ds', right_on='year', how='left')
    mundial.drop(columns=['year'], inplace=True, errors='ignore')
    mundial = mundial[mundial['ds'] >= 1985]
    mundial['ds'] = pd.to_datetime(mundial['ds'], format='%Y')

    changepoint_prior_scale = 0.1
    seasonality_prior_scale = 1.0
    seasonality_mode = 'additive'

    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode
    )

    prophet_df = mundial.copy()
    model.add_regressor('consumo')
    model.fit(prophet_df)

    last_date = prophet_df['ds'].max()
    years_to_predict = target_year - last_date.year

    future = model.make_future_dataframe(periods=years_to_predict, freq='Y')
    future = future.merge(
        prophet_df[['ds', 'consumo']],
        on='ds',
        how='left'
    )

    # Completar valores faltantes usando forward fill (último valor conocido)
    future['consumo'] = future['consumo'].fillna(method='ffill')
    forecast = model.predict(future)

    # Crear gráfico de la predicción mundial
    fig_mundial.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='lines+markers',
        name='Datos Históricos',
        line=dict(color='red', width=3)
    ))

    future_forecast = forecast[forecast['ds'] > last_date]

    if not future_forecast.empty:
        # Añadir línea que conecta los datos reales con la predicción
        fig_mundial.add_trace(go.Scatter(
            x=[prophet_df['ds'].iloc[-1], future_forecast['ds'].iloc[0]],
            y=[prophet_df['y'].iloc[-1], future_forecast['yhat'].iloc[0]],
            mode='lines+markers',
            line=dict(color='red', width=0.5),
            showlegend=False
        ))

        fig_mundial.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines+markers',
            name='Predicción',
            line=dict(color='blue', width=3)
        ))

    # Modelos por tipo de energía
    prophet_energy_df = pd.DataFrame()
    prophet_energy_df['ds'] = pd.to_datetime(produccion['year'], format='%Y')

    for c in columns:
        if c in produccion.columns and c in consumo.columns:
            energy_df = prophet_energy_df.copy()
            energy_df['y'] = produccion[c]
            energy_df['consumo'] = consumo[c]

            # Crear nuevo modelo en cada iteración
            energy_model = Prophet(
                yearly_seasonality=True,
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=1.0,
                seasonality_mode='additive'
            )

            energy_model.add_regressor('consumo')
            energy_model.fit(energy_df)

            energy_last_date = energy_df['ds'].max()
            energy_years_to_predict = target_year - energy_last_date.year

            energy_future = energy_model.make_future_dataframe(periods=energy_years_to_predict, freq='Y')
            energy_future = energy_future.merge(
                energy_df[['ds', 'consumo']],
                on='ds',
                how='left'
            )

            # Completar valores faltantes
            energy_future['consumo'] = energy_future['consumo'].fillna(method='ffill')
            energy_forecast = energy_model.predict(energy_future)

            # Obtener los colores para este tipo de energía
            historical_color = color_map.get(c, {'historical': 'sandybrown'})['historical']
            prediction_color = color_map.get(c, {'prediction': 'green'})['prediction']

            # Añadir al gráfico de energía con colores específicos sin etiquetas descriptivas
            fig_energia.add_trace(go.Scatter(
                x=energy_df['ds'],
                y=energy_df['y'],
                mode='lines+markers',
                name=f'{c}',
                line=dict(color=historical_color, width=2),
                legendgroup=c,
                showlegend=False
            ))

            energy_future_forecast = energy_forecast[energy_forecast['ds'] > energy_last_date]

            if not energy_future_forecast.empty:
                # Añadir línea que conecta los datos reales con la predicción
                fig_energia.add_trace(go.Scatter(
                    x=[energy_df['ds'].iloc[-1], energy_future_forecast['ds'].iloc[0]],
                    y=[energy_df['y'].iloc[-1], energy_future_forecast['yhat'].iloc[0]],
                    mode='lines+markers',
                    line=dict(color=prediction_color,  width=2),
                    showlegend=False,
                    legendgroup=c
                ))

                fig_energia.add_trace(go.Scatter(
                    x=energy_future_forecast['ds'],
                    y=energy_future_forecast['yhat'],
                    mode='lines+markers',
                    name=f'{c}',
                    line=dict(color=prediction_color, width=2.5),
                    legendgroup=c,
                    showlegend=True
                ))

    # Configuración de los layouts
    fig_mundial.update_layout(
        title="Predicción de Generación Total de Energía Mundial",
        xaxis=dict(title="Año", showgrid=True, griddash='dot'),
        yaxis=dict(title="Generación Total (TWh)", showgrid=True, griddash='dot'),
        legend=dict(x=0, y=1),
        template="plotly_white",
        width=1400,
        height=600
    )

    fig_energia.update_layout(
        title="Predicciones por Tipo de Energía",
        xaxis=dict(title="Año", showgrid=True, griddash='dot'),
        yaxis=dict(title="Generación (TWh)", showgrid=True, griddash='dot'),
        legend=dict(
            x=0.01,
            y=0.99,
            traceorder='grouped',
            groupclick='togglegroup'
        ),

        template="plotly_white",
        width=1400,
        height=600
    )

    return fig_mundial, fig_energia

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

    # Layout diferente según la región seleccionada
    if selected_region == "Mundial":
        # Para "Mundial", mostramos la tabla, el mapa y ambas predicciones
        col_tabla, col_predicciones = st.columns([1, 2])

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

        with col_predicciones:
            # Crear pestañas para mostrar los dos gráficos
            tab1, tab2 = st.tabs(["Predicción Global", "Predicción por Tipo de Energía"])

            with tab1:
                st.plotly_chart(fig_forecast, use_container_width=True)

            with tab2:
                # Generar predicciones por tipo de energía solo para Mundial
                with st.spinner("Generando predicciones por tipo de energía..."):
                    _, fig_energia = prophet_energias(data, prediction_year)
                    st.plotly_chart(fig_energia, use_container_width=True)
    else:
        # Para otras regiones, mantener el layout original
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

    # Mostrar distribución de energía (SIEMPRE AL FINAL)
    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Distribución de Energía</h2>",
               unsafe_allow_html=True)

    col_central = st.columns([1])[0]
    with col_central:
        fig_distribution = plot_energy_distribution(selected_region, data)
        st.plotly_chart(fig_distribution, use_container_width=True)

if __name__ == "__main__":
    main()
