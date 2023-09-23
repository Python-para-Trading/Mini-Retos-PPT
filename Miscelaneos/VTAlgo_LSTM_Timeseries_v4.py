# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:09:30 2023

@author: USER
"""

# Study Quant For Finance over LSTM Timeseries_v4
# VTAlgo Group de Elmer Niño. @Undertiker

# 1.- Start Script
import time
start_time = time.time()

# Importamos Librerías
# 2.1.- Importamos Librerías comunes
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
np.random.seed(4)
import matplotlib.pyplot as plt

# 2.2.- Importar Librerías sección de Machine Learning
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 3.- Data download
# 3.1.- Etiquetamos
project_name = "LTSM for Finance"
symbol = 'EURUSD=X'
start_date = '2010-01-01'

# 3.2.- Get the current date
today = datetime.date.today()

# 3.3.- Format the current date as a string in 'YYYY-MM-DD' format
end_date = today.strftime('%Y-%m-%d')

symbol_clean = symbol.replace('=X', '')
timeframe = "TF Diario"

# 3.4.- Define a label for titles and filenames
label = f"{project_name}; {symbol_clean}_{start_date} to {end_date}_{timeframe}"

# 4.- Download data
try:
    dataset = yf.download(symbol, start=start_date, end=end_date)
    if dataset.empty:
        print(f"No data found for {symbol} from {start_date} to {end_date}")
    else:
        print(dataset.head(3))
except Exception as e:
    print(f"An error occurred: {str(e)}")
    
# 4.- Indexamos mas features al dataset utilizando la librería ta y anexando manualmente el filtro de Kalman
# 4.1.- Importamos la Librería ta y hacemos una copia al dataset
import ta
dataset_ta = dataset.copy()

# 4.2.- Añadiendo media móvil simple
dataset_ta['SMA'] = ta.trend.sma_indicator(dataset_ta['Close'])

# 4.3.- Anexamos el filtro de Kalman
from pykalman import KalmanFilter
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kf = kf.em(dataset_ta['Close'], n_iter=10)
(filtered_state_means, _) = kf.filter(dataset_ta['Close'])
dataset_ta['Kalman Filter'] = filtered_state_means

# 4.4.- Asegúrate de manejar los valores NaN que puedan generar los indicadores técnicos
dataset_ta.dropna(inplace=True)

# 5.- Sets de entrenamiento y validación 
# La LSTM se entrenará con datos de 2016 hacia atrás. La validación se hará con datos de 2017 en adelante.
# 5.1.- Escogiendo el conjunto de entrenamiento del dataframe correcto con las columnas correctas
set_entrenamiento = dataset_ta[:'2016'].loc[:, ['High', 'SMA', 'Kalman Filter']]

# 5.2.- Escogiendo el conjunto de validación del dataframe correcto con las columnas correctas
set_validacion = dataset_ta['2017':].loc[:, ['High', 'SMA', 'Kalman Filter']]
set_entrenamiento['High'].plot(legend=True)
set_validacion['High'].plot(legend=True)
plt.legend(['Entrenamiento (2010-2016)', 'Validación (2017 hasta los presentes)'])
plt.show()

# 5.3.- Normalización del set de entrenamiento con nuevas características
sc = MinMaxScaler(feature_range=(0, 1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento[['High', 'SMA', 'Kalman Filter']])

# 5.4.- Utilizando el mismo objeto de escalado para transformar el set de validación
set_validacion_escalado = sc.transform(set_validacion)

# 5.5.- La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

# 5.6.- Ajustando el bucle para formar X_train y Y_train considerando todas las características
for i in range(time_step,m):
    X_train.append(set_entrenamiento_escalado[i-time_step:i, :])  # todas las características
    Y_train.append(set_entrenamiento_escalado[i, 0])  # la primera característica 'High'
X_train, Y_train = np.array(X_train), np.array(Y_train)

# 5.7.- Ajustar la forma de X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# 6.- Red LSTM
dim_entrada = (X_train.shape[1], X_train.shape[2])  # Cambiar la forma de entrada
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=20,batch_size=32)

# 7.- Validación (predicción del valor de las acciones)
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step, len(set_validacion_escalado)):
    X_test.append(set_validacion_escalado[i-time_step:i, :])
X_test = np.array(X_test)

prediccion = modelo.predict(X_test)

# 8.- Modificar cómo se hace la inversión de escalado para la predicción
nFeatures = 3
prediccion_transformed = np.zeros((len(prediccion), nFeatures))
prediccion_transformed[:,0] = prediccion[:,0]
prediccion_inverse = sc.inverse_transform(prediccion_transformed)[:,0]

# 9.- Graficar resultados con matplotlib
def graficar_predicciones(real, prediccion, titulo):
    plt.figure(figsize=(14,8))
    plt.plot(real[:len(prediccion)], color='red', label='Valor real del activo')
    plt.plot(prediccion, color='blue', label=f'Predicción de la acción con {titulo}')
    plt.title(titulo)
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor del activo')
    plt.legend()
    plt.show()

# Graficar 'High' con Predicción
graficar_predicciones(set_validacion['High'].values, prediccion_inverse, 'High')

# Graficar 'SMA' con Predicción
graficar_predicciones(set_validacion['SMA'].values, prediccion_inverse, 'SMA')

# Graficar 'Kalman Filter' con Predicción
graficar_predicciones(set_validacion['Kalman Filter'].values, prediccion_inverse, 'Kalman Filter')

# 10.- Graficamos con Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# 10.1.- Define una función para guardar la gráfica como HTML
def guardar_grafica(fig, filename):
    pio.write_html(fig, file=filename, auto_open=False)
    print(f"{filename} se ha guardado correctamente.")

# 10.2.- Asegurándote de que las fechas están en el formato correcto
# Verificar si el DataFrame tiene una columna 'Fecha'
if 'Fecha' not in set_validacion.columns:
    # Si el índice del DataFrame es un DatetimeIndex, crear una columna 'Fecha' a partir del índice
    if isinstance(set_validacion.index, pd.DatetimeIndex):
        set_validacion['Fecha'] = set_validacion.index.date
    else:
        raise ValueError("No hay una columna de fechas y el índice no es un DatetimeIndex.")

# 10.3.- Obtén las fechas desde tu set de validación
fechas = set_validacion['Fecha']

# 11.- Graficamos con Plotly y guardamos y visualizamos en un html
# 11.1.- Función Plotly
def graficar_predicciones_plotly(real, prediccion, titulo, fechas, label):
    fig = make_subplots(rows=1, cols=1)
    
    # Agregar traza para los valores reales
    fig.add_trace(
        go.Scatter(x=fechas[:len(prediccion)], y=real[:len(prediccion)], mode='lines', name='Valor real del activo', line=dict(color='white')),
        row=1, col=1
    )
    
    # Agregar traza para los valores predichos
    fig.add_trace(
        go.Scatter(x=fechas[:len(prediccion)], y=prediccion, mode='lines', name=f'Predicción de la acción con {titulo}', line=dict(color='cyan')),
        row=1, col=1
    )
    
    # Actualizar layout de la figura para tener fondo oscuro y grid tenue
    fig.update_layout(
        title=f"{label}, Features {titulo}",
        xaxis_title='Fecha',
        yaxis_title='Valor del activo',
        template='plotly_dark',
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)' # Grid casi transparente
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)' # Grid casi transparente
        )
    )
    
    # Guardar la figura como un archivo HTML
    filename = f"{titulo}_{label}_prediccion.html".replace(' ', '_').replace(';', '').replace(':', '')
    guardar_grafica(fig, filename)  # Utilizar la función definida anteriormente para guardar la gráfica
    
    return fig

# 11.2.- Obtén las fechas desde tu set de validación
fechas = set_validacion.index if set_validacion.index.name == 'Fecha' else set_validacion['Fecha']

# 11.3.- Graficar 'High' con Predicción
graficar_predicciones_plotly(set_validacion['High'].values, prediccion_inverse, 'High', fechas, label)

# 11.4.- Graficar 'SMA' con Predicción
graficar_predicciones_plotly(set_validacion['SMA'].values, prediccion_inverse, 'SMA', fechas, label)

# 11.5.- Graficar 'Kalman Filter' con Predicción
graficar_predicciones_plotly(set_validacion['Kalman Filter'].values, prediccion_inverse, 'Kalman Filter', fechas, label)

# 11.6.- Graficamos todos los features en un solo html
# 11.6.1.- Crear trazados para cada serie de datos
trace_real = go.Scatter(x=set_validacion.index[:len(prediccion_inverse)], y=set_validacion['High'].values[:len(prediccion_inverse)], mode='lines', name='Real (High)', line=dict(color='white'))
trace_pred = go.Scatter(x=set_validacion.index[:len(prediccion_inverse)], y=prediccion_inverse, mode='lines', name='Predicción (High)', line=dict(color='yellow'))
trace_sma = go.Scatter(x=set_validacion.index[:len(prediccion_inverse)], y=set_validacion['SMA'].values[:len(prediccion_inverse)], mode='lines', name='Real (SMA)', line=dict(color='cyan'))
trace_kalman = go.Scatter(x=set_validacion.index[:len(prediccion_inverse)], y=set_validacion['Kalman Filter'].values[:len(prediccion_inverse)], mode='lines', name='Real (Kalman Filter)', line=dict(color='magenta'))

# 11.6.2.- Crear layout personalizado
layout = go.Layout(
    title=f'Comparación de Predicciones, {label}',
    xaxis=dict(
        title='Fecha',
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=0.5,
        zerolinecolor='gray',
        zerolinewidth=0.5,
    ),
    yaxis=dict(
        title='Valor',
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=0.5,
        zerolinecolor='gray',
        zerolinewidth=0.5,
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# 11.6.3.- Crear figura y añadir trazados
fig = go.Figure(data=[trace_real, trace_pred, trace_sma, trace_kalman], layout=layout)

# 11.6.4.- Guardar la figura en un archivo HTML
fig.write_html(f'{label}_comparacion_predicciones.html')

# 12.- Cálculo de Métricas de Error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np  # Asegúrate de tener esta importación para usar np.mean y np.abs

# 12.1.- Para High
mse = mean_squared_error(set_validacion['High'].values[:len(prediccion_inverse)], prediccion_inverse)
mae = mean_absolute_error(set_validacion['High'].values[:len(prediccion_inverse)], prediccion_inverse)
mape = np.mean(np.abs((set_validacion['High'].values[:len(prediccion_inverse)] - prediccion_inverse) / set_validacion['High'].values[:len(prediccion_inverse)])) * 100

# 12.2.- Para 'SMA'
mse_sma = mean_squared_error(set_validacion['SMA'].values[:len(prediccion_inverse)], prediccion_inverse)
mae_sma = mean_absolute_error(set_validacion['SMA'].values[:len(prediccion_inverse)], prediccion_inverse)
mape_sma = np.mean(np.abs((set_validacion['SMA'].values[:len(prediccion_inverse)] - prediccion_inverse) / set_validacion['SMA'].values[:len(prediccion_inverse)])) * 100

# 12.3.- Para 'Kalman Filter'
mse_kalman = mean_squared_error(set_validacion['Kalman Filter'].values[:len(prediccion_inverse)], prediccion_inverse)
mae_kalman = mean_absolute_error(set_validacion['Kalman Filter'].values[:len(prediccion_inverse)], prediccion_inverse)
mape_kalman = np.mean(np.abs((set_validacion['Kalman Filter'].values[:len(prediccion_inverse)] - prediccion_inverse) / set_validacion['Kalman Filter'].values[:len(prediccion_inverse)])) * 100

# 12.4.- Imprimiendo los resultados:
print(f"Mean Squared Error (MSE) para 'High': {mse}")
print(f"Mean Absolute Error (MAE) para 'High': {mae}")
print(f"Mean Absolute Percentage Error (MAPE) para 'High': {mape}%")
print(f"Mean Squared Error (MSE) para 'SMA': {mse_sma}")
print(f"Mean Absolute Error (MAE) para 'SMA': {mae_sma}")
print(f"Mean Absolute Percentage Error (MAPE) para 'SMA': {mape_sma}%")
print(f"Mean Squared Error (MSE) para 'Kalman Filter': {mse_kalman}")
print(f"Mean Absolute Error (MAE) para 'Kalman Filter': {mae_kalman}")
print(f"Mean Absolute Percentage Error (MAPE) para 'Kalman Filter': {mape_kalman}%")

# 13.- Medición del tiempo total del Proceso Run del Script
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total de ejecución: {elapsed_time / 60:.2f} minutos")

# End Script