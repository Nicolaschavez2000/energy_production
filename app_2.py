import streamlit as st
import pandas as pd
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

data = pd.read_csv("World Energy Consumption.csv", index_col='year')
data.index = pd.to_datetime(data.index, format='%Y')

def load_prophet_data():
    africa_continente = pd.read_csv("continente_africa.csv")
    sudamerica_continente = pd.read_csv("continente_sudamerica.csv")
    oceania_continente = pd.read_csv("continente_oceania.csv")
    asia_continente = pd.read_csv("continente_asia.csv")

    def process_dataframe(df):
        df = df.set_index("Year").drop(columns=["Entity"])
        df.columns = ["electricity_generation"]
        df.index = pd.to_datetime(df.index, format='%Y')
        return df

    return {
        'Africa': (process_dataframe(africa_continente), 0.66),
        'Sud America': (process_dataframe(sudamerica_continente), 0.42),
        'Oceania': (process_dataframe(oceania_continente), 0.82),
        'Asia': (process_dataframe(asia_continente), 0.80)
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
    'Africa': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso',
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
    'Norte America': ['Canada', 'Greenland', 'Mexico', 'United States'],
    'Oceania': ['Australia', 'Fiji', 'New Zealand', 'Papua New Guinea'],
    'Sud America': ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',
                      'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']
}

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
    # Asignar continente a cada país según un diccionario (continent_map)
    data['continente'] = data['country'].map(
        lambda x: next((k for k, v in continent_map.items() if x in v), None)
    )
    df_continente = data[data['continente'] == continente]

    # Agrupar y filtrar datos (se asume que 'col_electricity' y 'rename_columns' ya están definidos)
    df_continente = df_continente.groupby(['year', 'continente'])[col_electricity].sum().reset_index('continente')
    df_continente = df_continente[df_continente.index.year > 1970]

    # Crear variables adicionales
    df_continente['carbon'] = df_continente['low_carbon_electricity'] + df_continente['coal_electricity']
    df_continente['renovable'] = df_continente['other_renewable_electricity'] + df_continente['other_renewable_exc_biofuel_electricity']
    df_continente.drop(
        columns=['low_carbon_electricity', 'other_renewable_electricity',
                 'other_renewable_exc_biofuel_electricity', 'coal_electricity'],
        inplace=True
    )
    df_continente.rename(columns=rename_columns, inplace=True)

    energy_sources = ['nuclear', 'petroleo', 'solar', 'eolica', 'fosil',
                        'gas', 'hidro', 'bio fuel', 'carbon', 'renovable']

    # Crear figura con dos subplots: uno para evolución (línea) y otro para distribución (donut)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Evolución de la producción de energía en {continente}',
                        f'Distribución Energía {continente}'),
        specs=[[{"type": "xy"}, {"type": "domain"}]]
    )

    # Agregar trazas para la evolución de la producción (línea)
    for source in energy_sources:
        fig.add_trace(
            go.Scatter(
                x=df_continente.index.year,
                y=df_continente[source],
                mode='lines+markers',
                name=source
            ),
            row=1, col=1
        )

    # Calcular totales y porcentajes para el gráfico de pastel
    totals = df_continente[energy_sources].sum()
    porcentajes = (totals / totals.sum()) * 100

    # Agregar la traza de pastel (donut)
    fig.add_trace(
        go.Pie(
            labels=totals.index,
            values=porcentajes,
            hole=0.5,
            textinfo='percent+label'
        ),
        row=1, col=2
    )

    # Ajustar el layout
    #fig.update_layout(margin=dict(t=50, b=0, l=0, r=0))
    fig.update_layout(width=1000,margin=dict(t=50, b=0, l=0, r=0))
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

    plt.xlim(country_filt.index[0], pd.Timestamp(f'{prediction_year}-01-01'))

    forecast_table = forecast[['ds', 'yhat']].copy()
    forecast_table['ds'] = forecast_table['ds'].dt.year
    forecast_table = forecast_table[(forecast_table['ds'] >= 2023) & (forecast_table['ds'] <= 2035)]
    forecast_table = forecast_table.round(2)
    forecast_table.columns = ['Año', 'Predicción']

    return fig, forecast_table

def main():
    st.title("Análisis y predicción de energia por continente")
    prophet_data = load_prophet_data()
    control_col1, control_col2 = st.columns(2)

    with control_col1:
        available_continents = list(set(continent_map.keys()) & set(prophet_data.keys()))
        continente = st.selectbox(
            "Selecciona un continente",
            [""] + available_continents
        )
    if continente:
        with control_col2:
            prediction_year = st.slider(
                'Seleccionar año de predicción:',
                min_value=2023,
                max_value=2035,
                value=2030,
                step=1
            )
        st.subheader("Distribución de Energía")
        fig_distribution = plot_energy_distribution(continente, data)
        st.plotly_chart(fig_distribution)

        st.subheader("Predicción de Generación Eléctrica")
        fig_forecast, forecast_data = plot_prophet_forecast(continente, prediction_year, prophet_data)
        st.pyplot(fig_forecast)

        # Capturar eventos de clic en el gráfico (especialmente en el pastel)
        selected = plotly_events(fig_distribution)
        if selected:
            fuente_seleccionada = selected[0].get('label')
            st.write(f"Has seleccionado: {fuente_seleccionada}")
        # Aquí se puede cargar o mostrar el análisis detallado según la fuente seleccionada


if __name__ == "__main__":
    main()


'''import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_energy_distribution(continente, data, df_dato):
    # ... tu código actual para crear fig_distribution ...

    for i, energia in enumerate(fig_distribution.data[0].labels):
        paises = df_dato[df_dato['energía'] == energia]['país'].tolist()
        fig_distribution.data[0].customdata = [paises] * len(fig_distribution.data[0].labels) # Repetimos la lista para cada sector del pie
        fig_distribution.data[0].hovertemplate = f"<b>{energia}</b><br>" + \
                                                "Países productores: %{customdata[0]}<extra></extra>"

    return fig_distribution'''
