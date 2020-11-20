# Mini Reto 006 del grupo Python para Trading

# Backtrader - Estrategia basada en cruce de medias moviles simples (SMA) 


Este mini reto es la continuación natural del Mini Reto 005 donde se calcularon las medias móviles rápidas y lentas para identificar los cruces y generar una señal a partir de dichos cruces.

En esta ocasión, lo que se pide es utilizar esa experiencia previa para montar una estrategia y hacer el backtesting para evaluar los resultados de la estrategia con datos de muestra.

Lo mas sencillo, que no lo más optimo ni "real", es hacer un backtest vectorizado, dónde se añaden columnas a cada periodo indicando el beneficio/perdida en ese periodo y finalmente obtener el acumulado para saber si ha habido o no beneficio. 

Esto ya se muestra en detalle en el siguiente Webinar ( https://www.youtube.com/watch?v=CNCRGcFh5Lo ), así que vamos a dar el siguiente paso utilizando `Backtrader` para simular la estrategia lo mas "real" posible. 

Partiendo del código que hay a continuación:

- 1) Diseñar la estrategia en base a los indicadores
- 2) Configurar el entorno de backtrader
- 3) Ejecutar el backtesting con la estrategia
- 4) Analizar los resultados


**Se pide** COMPLETAR el punto 1 para hacer funcionar el Notebook al completo.