from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo, Blackly

import warnings
warnings.filterwarnings('ignore')

import backtrader as bt
import yfinance as yf

from math import floor

## Estrategia de cruce de medias con porcentaje de exposición

# Clase heredada de bt.Sizer al que le pasamos el porcentaje 
# que queremos estar expuestos

class PercentValue(bt.Sizer):
    params = (('percent', .90),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        # Calculamos la posición diviendo el porcentaje deseado 
        # del valor de la cartera entre el precio del activo
        return floor((self.broker.get_value() * self.params.percent / data) )


## Diseñamos la estrategia

class cruceSMA_(bt.Strategy):
    params = dict(slow=14,
                  fast=60
                 )

    def __init__(self):
        fast = bt.ind.SMA(period=self.p.fast)
        slow = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(fast, slow)

    def next(self):
        
        
        if self.crossover != 0 and self.position:  # Si hay una operacion abierta y ocurre cruce 
            self.close()                          # cerramos la posición, para después abrir contraria
        if self.crossover == 1: # si la rapida cruza hacia arriba la lenta
            self.buy()         # Se entra en LONG
        if self.crossover == -1: # si la rapida cruza hacia abajo la lenta
            self.sell()        # Se entra en SHORT



if __name__ == '__main__':


    ## Definimos los parametros de la estrategia

    fast_SMA = 9
    slow_SMA = 30
    cash = 1000
    exposure=.90
    
    # Descargamos los datos

    aapl = yf.download('AAPL', '2020-1-1','2020-10-25')


    ## Configuramos el entorno

    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(cash)

    data = bt.feeds.PandasData(dataname=aapl)
    cerebro.adddata(data)

    cerebro.addstrategy(cruceSMA_, slow=slow_SMA, fast=fast_SMA) # Determinamos la estrategia
    cerebro.addsizer(PercentValue, percent=exposure)  # Añadimos el sizer
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')  # Añadimos el analizador


    ## Ejecutamos el backtesting

    results = cerebro.run()

    b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
    cerebro.plot(b)