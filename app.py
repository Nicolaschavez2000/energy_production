import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análisis de Energía",
    layout="wide",
    initial_sidebar_state="collapsed"
)

data = pd.read_csv("World Energy Consumption.csv", index_col='year')
data.index = pd.to_datetime(data.index, format='%Y')

def load_prophet_data():
    africa_continente = pd.read_csv("continente_africa.csv")
    sudamerica_continente = pd.read_csv("continente_sudamerica.csv")
    oceania_continente = pd.read_csv("continente_oceania.csv")
    asia_continente = pd.read_csv("continente_asia.csv")
    norteamerica_continente = pd.read_csv("continente_norteamerica.csv")
    europa_continente = pd.read_csv("continente_europa.csv")
    mundial = pd.read_csv("world_generation.csv")

    def process_dataframe(df):
        df = df.set_index("Year")
        df.columns = ["electricity_generation"]
        df.index = pd.to_datetime(df.index, format='%Y')
        return df

    return {
        'África': (process_dataframe(africa_continente), 0.66),
        'Sud América': (process_dataframe(sudamerica_continente), 0.42),
        'Oceanía': (process_dataframe(oceania_continente), 0.82),
        'Asia': (process_dataframe(asia_continente), 0.80),
        'Norte América': (process_dataframe(norteamerica_continente), 0.7),
        'Europa': (process_dataframe(europa_continente), 0.7),
        'Mundial': (process_dataframe(mundial), 0.7)
    }

col_electricity = ['low_carbon_electricity',
            'nuclear_electricity',
            'oil_electricity',
            'other_renewable_electricity',
            'other_renewable_exc_biofuel_electricity',
            'solar_electricity',
            'wind_electricity',
            'fossil_electricity',
            'gas_electricity',
            'hydro_electricity',
            'biofuel_electricity',
            'coal_electricity']

continent_map = {
    'África': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso',
               'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic',
               'Chad', 'Comoros', 'Congo', 'Cote d\'Ivoire', 'Democratic Republic of the Congo',
               'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini',
               'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya',
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

continent_map['Mundial'] = []
for continent, countries in continent_map.items():
    if continent != 'Mundial':
        continent_map['Mundial'].extend(countries)

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

def plot_energy_distribution(continente, data):
    if continente == 'Mundial':
        df_continente = data.groupby(['year'])[col_electricity].sum()
    else:
        data['continente'] = data['country'].map(lambda x: next((k for k, v in continent_map.items() if x in v), None))
        df_continente = data[data['continente'] == continente]
        df_continente = df_continente.groupby(['year'])[col_electricity].sum()

    df_continente = df_continente[df_continente.index.year > 1970]

    df_continente['carbon'] = df_continente['low_carbon_electricity'] + df_continente['coal_electricity']
    df_continente['renovable'] = df_continente['other_renewable_electricity'] + df_continente['other_renewable_exc_biofuel_electricity']

    df_continente.drop(columns=['low_carbon_electricity', 'other_renewable_electricity',
                              'other_renewable_exc_biofuel_electricity', 'coal_electricity'],
                      inplace=True)
    df_continente.rename(columns=rename_columns, inplace=True)

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette('muted', 10)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    energy_sources = ['nuclear', 'petroleo', 'solar', 'eolica', 'fosil',
                     'gas', 'hidro', 'bio fuel', 'carbon', 'renovable']

    sns.lineplot(
        data=df_continente[energy_sources],
        palette=colors,
        linestyle='-',
        markers=True,
        dashes=False,
        ax=ax[0],
        linewidth=2.5
    )

    ax[0].set_title(f'Evolución de la producción de energía en {continente}',
                    fontsize=16, weight='bold', color='#333333')
    ax[0].set_xlabel('Año', fontsize=14)
    ax[0].set_ylabel('Energía (TW)', fontsize=14)
    ax[0].legend(energy_sources, loc='upper left',
                bbox_to_anchor=(0.05, 0.95), fontsize=12,
                frameon=True, shadow=True)
    ax[0].grid(True, linestyle='--', alpha=0.5)

    totals = df_continente[energy_sources].sum()
    porcentajes = (totals / totals.sum()) * 100

    patches, texts, autotexts = ax[1].pie(
        porcentajes,
        labels=totals.index,
        colors=colors,
        autopct=lambda p: '{:.1f}%'.format(p) if p > 0 else '',
        textprops={'color': "black", 'fontsize': 12},
        pctdistance=0.85,
        startangle=140,
        wedgeprops=dict(edgecolor='w')
    )

    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax[1].add_artist(centre_circle)

    ax[1].set_title(f'Distribución Energía {continente}',
                    fontsize=16, weight='bold', color='#333333')

    plt.tight_layout()
    return fig

def plot_prophet_forecast(continente, prediction_year, data_dict):
    country_filt, best_size = data_dict[continente]

    index = round(best_size * country_filt.shape[0])
    country_train = country_filt.iloc[:index]
    country_test = country_filt.iloc[index:]

    df_prophet_country = country_train.reset_index().rename(columns={
        "Year": "ds",
        "electricity_generation": "y"
    })

    model_prophet = Prophet()
    model_prophet.fit(df_prophet_country)

    last_date = country_filt.index[-1]
    years_to_predict = prediction_year - last_date.year
    total_periods = country_test.shape[0] + years_to_predict

    future = model_prophet.make_future_dataframe(periods=total_periods, freq='Y')

    forecast = model_prophet.predict(future)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(forecast['ds'], forecast['yhat'], color='blue', label='Predicción')
    ax.fill_between(forecast['ds'],
                   forecast['yhat_lower'],
                   forecast['yhat_upper'],
                   color='blue',
                   alpha=0.2,
                   label='Intervalo de incertidumbre')

    ax.scatter(country_train.index,
              country_train['electricity_generation'],
              color='black',
              s=20,
              label='Datos observados')

    ax.scatter(country_test.index,
              country_test['electricity_generation'],
              color='red',
              s=20,
              label='Datos reales')

    plt.title(f"Predicción de Generación Eléctrica - {continente}")
    plt.xlabel("Año")
    plt.ylabel("Generación Eléctrica (TWh)")
    plt.grid(True)
    plt.legend()

    plt.xlim(country_filt.index[0], pd.Timestamp(f'{prediction_year}-12-31'))

    forecast_table = forecast[['ds', 'yhat']].copy()
    forecast_table['ds'] = forecast_table['ds'].dt.year
    forecast_table = forecast_table[(forecast_table['ds'] >= 2023) & (forecast_table['ds'] <= prediction_year)]

    forecast_table['ds'] = forecast_table['ds'].apply(lambda x: f"{x + 1:,}".replace(",", ""))

    forecast_table['yhat'] = forecast_table['yhat'].apply(lambda x: f"{x:.2f}".replace(".", ","))

    forecast_table.columns = ['Año', 'Predicción - TWh']

    return fig, forecast_table

def plot_continent_map(selected_continent):
    continent_countries = {
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
                       'PER', 'SUR', 'URY', 'VEN']
    }

    fig = go.Figure()

    if selected_continent == 'Mundial':
        all_countries = []
        for countries in continent_countries.values():
            all_countries.extend(countries)

        fig.add_trace(go.Choropleth(
            locations=all_countries,
            z=[1] * len(all_countries),
            colorscale=[[0, 'rgb(200, 200, 255)'], [1, 'rgb(200, 200, 255)']],
            showscale=False,
            marker_line_color='rgb(100, 100, 100)',
            marker_line_width=0.5
        ))
    else:
        countries = continent_countries.get(selected_continent, [])
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
    st.title("Análisis y predicción de energia por continente")

    prophet_data = load_prophet_data()

    control_col1, control_col2, control_col3 = st.columns([1, 1, 2])

    with control_col1:
        available_continents = list(set(continent_map.keys()) & set(prophet_data.keys()))
        continente = st.selectbox(
            "Selecciona un continente",
            available_continents,
            index=available_continents.index("Mundial")
        )

    with control_col2:
        prediction_year = st.slider(
            'Seleccionar año de predicción:',
            min_value=2024,
            max_value=2035,
            value=2030,
            step=1
        )

    fig_forecast, forecast_data = plot_prophet_forecast(continente, prediction_year, prophet_data)

    st.subheader("Predicción de Generación Eléctrica")
    col_tabla, col_prediccion = st.columns([1, 2])

    with col_tabla:
        st.markdown("### Valores Predichos")
        st.dataframe(
            forecast_data,
            hide_index=True,
            use_container_width=True
        )

        fig_map = plot_continent_map(continente)
        st.plotly_chart(fig_map, use_container_width=True)

    with col_prediccion:
        st.pyplot(fig_forecast, use_container_width=True)

    st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Distribución de Energía</h2>", unsafe_allow_html=True)

    col_central = st.columns([1])[0]
    with col_central:
        fig_distribution = plot_energy_distribution(continente, data)
        st.pyplot(fig_distribution, use_container_width=True)

if __name__ == "__main__":
    main()
