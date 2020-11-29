# Mini Reto 007 del grupo Python para Trading

## Backtrader - Estrategia de reversión a la media RSI 25/75 


Trás el Mini-Reto 006 continuamos con Backtrader, implementando una estrategia bien conocida que incrementa un poco su complejidad.

Se trata de la estrategia RSI 25/75 publicada por Larry Connors y Cesar Álvarez en su libro “High Probability ETF Trading: 7 Professional Strategies to Improve Your ETF Trading”. 

Utiliza el índice de fuerza relativa RSI para medir cuando un activo se encuentra sobrevendido durante una tendencia alcista o sobrecomprado durante una tendencia bajista. Sugieren bajar el período de tiempo para el indicador RSI de su nivel habitual de 14 a 4, para aumentar significativamente el número de operaciones, con un indicador más "nervioso".

El sistema utiliza una de media móvil simple, con un periodo de 200, para determinar la tendencia a largo plazo. 

En mercado con tendencia alcista, es decir precio de cierre por encima de la media de 200 días, se abre posición larga cuando el indicador RSI cae por debajo 25. Se cierra la posición cuando el RSI cruza por encima de 55. 

En mercado bajista, abre posición corta cuando el RSI cruza por encima 75 y cierra cuando el RSI cae por debajo 45.


Para hacerlo más cercano a la realidad debes incorporar una comisión de 0.0035 USD por acción.

Y como gestión de capital invertir el 90% del valor de la cartera en cada operación, mediante un sizer de Backtrader.



Partiendo del código que hay a continuación:

- 1) Diseñar la estrategia en base a los indicadores
- 2) Configurar el entorno de backtrader
- 3) Ejecutar el backtest con la estrategia
- 4) Optimizar los parámetros de periodos del RSI y SMA para maximizar el ratio de Sharpe.
- 5) Ejecutar el backtest con los parámetros obtenidos
- 6) Analizar los resultados con Pyfolio
