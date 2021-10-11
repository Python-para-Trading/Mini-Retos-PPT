# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:48:04 2021

@author: J3

Comentarios:
    
    Fuente de datos https://finnhub.io/docs/api/financials-reported
    
    Pagina interesante : https://finviz.com/quote.ashx?t=app#statements
    
    PROBADO Y FUNCIONA BIEN

"""
from pandas.io.json import json_normalize
import pandas as pd
from fhub import Session
import json

import fhub as fn


# 0.- Debemos abrirnos una cuenta en FINNHUB
hub = Session("pon tu clave ")   

# Download prices time serie of Tesla.
#tsla = hub.candle('TSLA')

# Download prices for several tickers from a date.
#data = hub.candle(['AMZN', 'NFLX', 'DIS'], start="2018-01-01")

# Download prices and bollinger bands indicator for several tickers.
#data = hub.indicator(['AAPL', 'MSFT'], start='2019-01-01', indicator='bbands',
#                 indicator_fields={'timeperiod': 10})


# 1.- Recogemos todos los reportes finacieros de un simbolo
data3=hub.financials_as_reported(symbol='TSLA', freq='annual')

# 2.- Filtramos la base de datos por año concreto 
aux_01 = data3.loc[:, 'year'] == 2020
data4 = data3.loc[aux_01]
print(aux_01)
print(data4.head())

# 3.- Ahora tenemos una df con datos formateados en JSON. Tenemos que ordenar el contenido para encontrar los datos fundamentales
json_struct = json.loads(data4.to_json(orient="records"))
incomeStatement = pd.json_normalize(json_struct, 'ic')
balanceSheet = pd.json_normalize(json_struct, 'bs')
CashFlow = pd.json_normalize(json_struct, 'cf')

#   4.- datos ordenados: tenemos en los tres dataframe (income... balance... cashf...) 
#   los datos ordenados para un año



# 5.- Busco el campo desado. Ojo, que los literales de los campos no son constantes.
datoConcreto = incomeStatement.loc[incomeStatement['concept'].str.fullmatch("Revenues", case=False)]

# 6.- A partir de aquí ya tengo los datos ordenados en el dataframe. Busco el dato y lo trabajo


