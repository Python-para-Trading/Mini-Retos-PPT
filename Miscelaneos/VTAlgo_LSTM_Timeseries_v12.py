# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:27:48 2023

@author: USER
"""

# Study Quant For Finance over LSTM Timeseries_v11
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

# 2.2.- Importar Librerías de AT y Gráficos
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# 2.3.- Importar Librerías sección de Machine Learning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
from tensorflow.keras import regularizers

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
    
print(dataset[['Close']].describe())  # Esto te dará estadísticas descriptivas de la columna 'High'.
print(dataset)
    
# 4.- Indexamos mas features al dataset utilizando la librería ta y anexando manualmente el filtro de Kalman
# 4.1.- Hacemos una copia al dataset
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
set_entrenamiento = dataset_ta[:'2019'].loc[:, ['Close', 'SMA', 'Kalman Filter']]

# 5.2.- Escogiendo el conjunto de validación del dataframe correcto con las columnas correctas
set_validacion = dataset_ta['2020':].loc[:, ['Close', 'SMA', 'Kalman Filter']]
set_entrenamiento['Close'].plot(legend=True)
set_validacion['Close'].plot(legend=True)
plt.legend(['Entrenamiento (2010-2019)', 'Validación (2020 hasta los presentes)'])
plt.show()

# 5.3.- Normalización del set de entrenamiento con nuevas características
sc = MinMaxScaler(feature_range=(0, 1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento[['Close', 'SMA', 'Kalman Filter']])

# 5.4.- Utilizando el mismo objeto de escalado para transformar el set de validación
set_validacion_escalado = sc.transform(set_validacion)

# 5.5.- La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

# 5.6.- Ajustando el bucle para formar X_train y Y_train considerando todas las características
for i in range(time_step, m):
    X_train.append(set_entrenamiento_escalado[i-time_step:i])  # todas las características
    Y_train.append(set_entrenamiento_escalado[i, :])  # todas las características ahora, no solo 'High'

X_train, Y_train = np.array(X_train), np.array(Y_train)

# 5.7.- Ajustar la forma de X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

# 6.- Construir la estructura de la red LSTM
dim_entrada = (X_train.shape[1], X_train.shape[2])
dim_salida = X_train.shape[2]  # la salida ahora considera todas las características
neuronas = 50

modelo = Sequential()
modelo.add(LSTM(units=neuronas, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.summary()

# 7.- Entrenar el modelo LSTM
epocas = 100
batch_size = 32
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

history = modelo.fit(
    X_train, Y_train,
    epochs=epocas,
    batch_size=batch_size,
    validation_split=0.2,  # suponiendo que quieres utilizar un 20% de tus datos para validación
    callbacks=[early_stop]
)

# 8.- Predicción de los precios
set_validacion = dataset_ta['2020':].loc[:, ['Close', 'SMA', 'Kalman Filter']]
total_dataset = pd.concat((set_entrenamiento[['Close', 'SMA', 'Kalman Filter']], set_validacion), axis=0)

inputs = total_dataset[len(total_dataset) - len(set_validacion) - time_step:].values
inputs = inputs.reshape(-1,3)
inputs = sc.transform(inputs)

X_test = []
for i in range(time_step, time_step + len(set_validacion)):
    X_test.append(inputs[i-time_step:i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

predicted_values = modelo.predict(X_test)
predicted_values = sc.inverse_transform(predicted_values)

# 8.1.- Después de predecir con el modelo LSTM y antes de graficar
df_predicciones = pd.DataFrame(predicted_values, columns=['Close', 'SMA', 'Kalman Filter'], index=set_validacion.index[:len(predicted_values)])

# 8.5.- Predicciones "out-of-sample"

# Tomar los últimos time_step puntos de datos
last_time_step = total_dataset[-time_step:].values
last_time_step = last_time_step.reshape(-1, 3)
last_time_step = sc.transform(last_time_step)

X_last = []
X_last.append(last_time_step)
X_last = np.array(X_last)
X_last = np.reshape(X_last, (X_last.shape[0], X_last.shape[1], X_last.shape[2]))

# Predicción del próximo punto en la serie temporal
predicted_next_value = modelo.predict(X_last)
predicted_next_value = sc.inverse_transform(predicted_next_value)
print("Predicción del próximo punto:", predicted_next_value)

# Predicciones múltiples días en el futuro
days_to_predict = 5
predicted_values = []

for _ in range(days_to_predict):
    predicted_next_value = modelo.predict(X_last)
    predicted_next_value_rescaled = sc.inverse_transform(predicted_next_value)
    
    predicted_values.append(predicted_next_value_rescaled[0])
    
    new_point = predicted_next_value[0]
    X_last = np.append(X_last[:, 1:, :], [[new_point]], axis=1)

predicted_values = np.array(predicted_values)
print(predicted_values)
print(f"Predicciones para los próximos {days_to_predict} días:", predicted_values)

# 9.- Graficamos con Plotly
# 9.1.- Define una función para guardar la gráfica como HTML
def guardar_grafica(fig, filename):
    pio.write_html(fig, file=filename, auto_open=False)
    print(f"{filename} se ha guardado correctamente.")

# 9.2.- Asegurándote de que las fechas están en el formato correcto
# Verificar si el DataFrame tiene una columna 'Fecha'
if 'Fecha' not in set_validacion.columns:
    # Si el índice del DataFrame es un DatetimeIndex, crear una columna 'Fecha' a partir del índice
    if isinstance(set_validacion.index, pd.DatetimeIndex):
        set_validacion['Fecha'] = set_validacion.index.date
    else:
        raise ValueError("No hay una columna de fechas y el índice no es un DatetimeIndex.")

# 9.3.- Obtén las fechas desde tu set de validación
fechas = set_validacion['Fecha']
# Asumiendo que la última fecha en tus datos es la última fecha en set_validacion
last_date = set_validacion.index[-1]
future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, days_to_predict + 1)]
extended_fechas = set_validacion.index.tolist() + [set_validacion.index[-1] + pd.DateOffset(days=i) for i in range(1, days_to_predict + 1)]

# 10.- Graficamos con Plotly y guardamos y visualizamos en un html
# 10.1.- Función Plotly
def graficar_predicciones_plotly(real, prediccion, out_of_sample_pred, titulo, extended_fechas, future_dates, label):
    fig = make_subplots(rows=1, cols=1)
    
    # Datos reales
    fig.add_trace(
        go.Scatter(x=extended_fechas[:-len(out_of_sample_pred)], y=real, mode='lines', name='Valor real del activo', line=dict(color='white'))
    )
    
    # Predicciones dentro de la muestra
    fig.add_trace(
        go.Scatter(x=extended_fechas[-len(prediccion):], y=prediccion, mode='lines', name=f'Predicción de la acción con {titulo}', line=dict(color='cyan'))
    )
    
    # Predicciones "out-of-sample"
    fig.add_trace(
        go.Scatter(x=future_dates, y=out_of_sample_pred, mode='lines', name=f'Predicción "out-of-sample" con {titulo}', line=dict(color='yellow'))
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
    
    fig.show()

    return fig

# 10.2.- Obtén las fechas desde tu set de validación
# fechas = set_validacion.index if set_validacion.index.name == 'Fecha' else set_validacion['Fecha']

# 10.2.- Extiende tus fechas para incluir las fechas de tus predicciones "out-of-sample"
extended_fechas = set_validacion.index.tolist() + [set_validacion.index[-1] + pd.DateOffset(days=i) for i in range(1, days_to_predict + 1)]

# 10.3.- Graficar 'Close' con Predicción
graficar_predicciones_plotly(dataset_ta['Close'].values, df_predicciones['Close'].values, predicted_values[:, 0], 'Close', extended_fechas, future_dates, label)

# 10.4.- Graficar 'SMA' con Predicción
graficar_predicciones_plotly(dataset_ta['SMA'].values, df_predicciones['SMA'].values, predicted_values[:, 1], 'SMA', extended_fechas, future_dates, label)

# 10.5.- Graficar 'Kalman Filter' con Predicción
graficar_predicciones_plotly(dataset_ta['Kalman Filter'].values, df_predicciones['Kalman Filter'].values, predicted_values[:, 2], 'Kalman Filter', extended_fechas, future_dates, label)

# 10.6.- Graficamos todos los features en un solo html
# 10.6.1.- Agregar traza para 'High' histórico y predicciones futuras
valores_close_totales = set_validacion['Close'].values.tolist() + df_predicciones['Close'].tolist()
trace_close = go.Scatter(x=fechas, y=valores_close_totales, mode='lines', name='Pred. (Close)', line=dict(color='white'))

# 10.6.2.- Agregar traza para 'SMA' histórico y predicciones futuras
valores_sma_totales = set_validacion['SMA'].values.tolist() + df_predicciones['SMA'].tolist()
trace_sma = go.Scatter(x=fechas, y=valores_sma_totales, mode='lines', name='Pred. (SMA)', line=dict(color='cyan'))

# 10.6.3.- Agregar traza para 'Kalman Filter' histórico y predicciones futuras
valores_kalman_totales = set_validacion['Kalman Filter'].values.tolist() + df_predicciones['Kalman Filter'].tolist()
trace_kalman = go.Scatter(x=fechas, y=valores_kalman_totales, mode='lines', name='Pred. (Kalman Filter)', line=dict(color='magenta'))

# 10.7.- Crear layout personalizado
# Ajustar el rango del eje Y basándose en los valores de 'High'.
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
        zerolinewidth=0.5
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white'),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# 10.8.- Crear figura y añadir trazados
fig = go.Figure(data=[trace_close, trace_sma, trace_kalman], layout=layout)
fig.show()

# 10.9.- Guardar la figura en un archivo HTML
fig.write_html(f'{label}_comparacion_predicciones.html')

# 11.- Cálculo de Métricas de Error
# 11.1.- Asumiendo que has creado el DataFrame 'df_predicciones' como mencioné anteriormente:
prediccion_close = df_predicciones['Close'].values
prediccion_sma = df_predicciones['SMA'].values
prediccion_kalman = df_predicciones['Kalman Filter'].values

# 11.2.- Para 'Close'
mse_close = mean_squared_error(set_validacion['Close'].values[:len(prediccion_close)], prediccion_close)
mae_close = mean_absolute_error(set_validacion['Close'].values[:len(prediccion_close)], prediccion_close)
mape_close = np.mean(np.abs((set_validacion['Close'].values[:len(prediccion_close)] - prediccion_close) / set_validacion['Close'].values[:len(prediccion_close)])) * 100

# 11.3.- Para 'SMA'
mse_sma = mean_squared_error(set_validacion['SMA'].values[:len(prediccion_sma)], prediccion_sma)
mae_sma = mean_absolute_error(set_validacion['SMA'].values[:len(prediccion_sma)], prediccion_sma)
mape_sma = np.mean(np.abs((set_validacion['SMA'].values[:len(prediccion_sma)] - prediccion_sma) / set_validacion['SMA'].values[:len(prediccion_sma)])) * 100

# 11.4.- Para 'Kalman Filter'
mse_kalman = mean_squared_error(set_validacion['Kalman Filter'].values[:len(prediccion_kalman)], prediccion_kalman)
mae_kalman = mean_absolute_error(set_validacion['Kalman Filter'].values[:len(prediccion_kalman)], prediccion_kalman)
mape_kalman = np.mean(np.abs((set_validacion['Kalman Filter'].values[:len(prediccion_kalman)] - prediccion_kalman) / set_validacion['Kalman Filter'].values[:len(prediccion_kalman)])) * 100

# 12.- Imprimiendo los resultados:
print(f"Mean Squared Error (MSE) para 'Close': {mse_close}")
print(f"Mean Absolute Error (MAE) para 'Close': {mae_close}")
print(f"Mean Absolute Percentage Error (MAPE) para 'Close': {mape_close}%")
print(f"Mean Squared Error (MSE) para 'SMA': {mse_sma}")
print(f"Mean Absolute Error (MAE) para 'SMA': {mae_sma}")
print(f"Mean Absolute Percentage Error (MAPE) para 'SMA': {mape_sma}%")
print(f"Mean Squared Error (MSE) para 'Kalman Filter': {mse_kalman}")
print(f"Mean Absolute Error (MAE) para 'Kalman Filter': {mae_kalman}")
print(f"Mean Absolute Percentage Error (MAPE) para 'Kalman Filter': {mape_kalman}%")

# 13.- Medición del tiempo total del Proceso Run del Script
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")

# End Script