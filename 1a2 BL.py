import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import risk_models, expected_returns
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

# Rango de fechas
inicio = '2003-01-03'
fin = '2020-12-30'

# Tickers de los activos
tickers = [    "XOM","GE","MSFT","C","BAC","WMT","PG","PFE","JNJ","AIG",
        "IBM","CVX","INTC","WFC","T","KO","VZ","PEP","HPQ","HD","AMGN",
        "UPS","COP","QCOM","MRK","UNH","ORCL","ABT","AXP","JPM","MO",
        "GS","MS","LLY","MDT","BA","CMCSA","MMM","SLB","CSCO","CL",
        "FNMA","VLO","CAT","TGT","DE","OXY","DD","FDX","LOW","CVS",
        "PRU","HAL","EXC","STT","COF","LMT","ALL","KR","DUK","MET",
        "COST","EMR","SBUX","CI","PNC","PGR","ITW","TXT","SO","HON",
        "PPG","BSX","MMC","NOC","EOG","DVN","NUE","FCX","USB","NSC",
        "UNP","KEY","BAX","AZO","GPC","DHR","BEN","TMO","TROW","BXP",
        "SPG","KMB","SCHW","RF","WMB","PAYX", "D", "GOLD", "TLT"]

# Descargamos datos
activos = yf.download(tickers, start = inicio, end = fin, auto_adjust=False)
activos = activos.loc[:,('Adj Close', slice(None))]
activos.columns = tickers


# Configuración de rebalanceo

ventana_meses = 36
fechas_rebalanceo = activos.resample("ME").last().index[ventana_meses:]

view_periods = {
    # Sector energía tuvo el mayor CAGR a 3 años en 2005
    (2006, 2008): ["XOM", "CVX", "COP", "SLB", "OXY"], 
    # Sector consumo básico tuvo el mayor CAGR a 3 años en 2008
    (2009, 2011): ["WMT", "MO", "PG", "KO", "PEP"],
    # Sector salud tuvo el mayor CAGR a 3 años en 2011
    (2012, 2014): ["JNJ", "AMGN", "DHR", "BAX", "BSX"],
    # Sector salud volvió a tener el mayor CAGR a 3 años en 2014
    (2015, 2017): ["JNJ", "AMGN", "DHR", "BAX", "BSX"],
    # Sector tecnología tuvo el mayor CAGR a 3 años en 2005
    (2018, 2020): ["MSFT", "INTC", "ORCL", "CSCO", "IBM"],
}

# Bucle de rebalanceo mensual
ponderaciones_fecha = []

for fecha in fechas_rebalanceo:
    start = fecha - pd.DateOffset(months = ventana_meses)
    precios_ventana = activos.loc[start:fecha].dropna(axis=1)

    if precios_ventana.shape[1] < 2:
        continue

    # Covarianza
    cov_matrix = risk_models.sample_cov(precios_ventana)

    # Prior
    pi = expected_returns.mean_historical_return(precios_ventana)

    # Determinar views activas
    year = fecha.year
    view_tickers = []
    for (y1, y2), tickers in view_periods.items():
        if y1 <= year < y2:
            view_tickers = [t for t in tickers if t in precios_ventana.columns]
            break

    if not view_tickers:
        continue

    # Defino mi confianza en las views y el % de rentabilidad que espero de las empresas seleccionadas
    views_absolutas = {ticker: 0.1 for ticker in view_tickers}
    confianza_en_views = [0.6] * len(views_absolutas)

    # Modelo Black-Litterman
    bl = BlackLittermanModel(
        cov_matrix,
        pi=pi,
        absolute_views = views_absolutas,
        view_confidences = confianza_en_views,
        omega="idzorek",
    )

    bl_retornos = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # Optimización con restricción del 10 % por activo
    ef = EfficientFrontier(bl_retornos, bl_cov, weight_bounds=(0, 0.10))
    weights = ef.max_sharpe()
    # Limpiamos los pesos redondeando
    pesos_limpios = ef.clean_weights()

    # guardamos los resultados
    ponderaciones_fecha.append({
        "fecha": fecha,
        "ponderaciones": pesos_limpios
    })


def backtest_BL(retornos, ponderaciones_fecha):
    rentabilidad_cartera = []
    fechas = retornos.index
    rebalance_dict = {r["fecha"]: r["ponderaciones"] for r in ponderaciones_fecha}
    pesos = None

    for fecha in fechas:
        if fecha in rebalance_dict:
            pesos = rebalance_dict[fecha]

        if pesos is not None:
            activos_disponibles = [t for t in pesos if t in retornos.columns]
            peso_vector = np.array([pesos[t] for t in activos_disponibles])
            retorno_vector = retornos.loc[fecha, activos_disponibles]
            retorno_port = np.sum(peso_vector * retorno_vector)
            rentabilidad_cartera.append(retorno_port)

    fechas_validas = fechas[-len(rentabilidad_cartera):]
    return pd.Series(rentabilidad_cartera, index=fechas_validas)





retornos_diarios = activos.pct_change().dropna()
rentabilidad_cartera = backtest_BL(retornos_diarios, ponderaciones_fecha)


print("\n--- Métricas de la cartera de Black-litterman ---")
dias_cotizacion_al_año = 252

# 1. Rentabilidad anual bruta 
print(f"Rentabilidad anual bruta: {rentabilidad_cartera.mean() * dias_cotizacion_al_año:.2%}")

# 2. CAGR (Compound Annual Growth Rate)
print(f"CAGR: {((1 + rentabilidad_cartera).cumprod().iloc[-1])**(1 / (len(rentabilidad_cartera) / dias_cotizacion_al_año)) - 1:.2%}")

# 3. Volatilidad anualizada
print(f"Volatilidad anualizada: {rentabilidad_cartera.std() * np.sqrt(dias_cotizacion_al_año):.2%}")

# 4. Ratio de Sharpe asumiento rf = 0
rentabilidad_anual_simple_sharpe = rentabilidad_cartera.mean() * dias_cotizacion_al_año
volatilidad_anualizada_sharpe = rentabilidad_cartera.std() * np.sqrt(dias_cotizacion_al_año)
print(f"Ratio de Sharpe: {rentabilidad_anual_simple_sharpe / volatilidad_anualizada_sharpe:.2f}")

# Para Max drawdown y Ulcer Index, calculamos los retornos acumulados una vez
retornos_acumulados_mdd_ulcer = (1 + rentabilidad_cartera).cumprod()
pico_mdd_ulcer = retornos_acumulados_mdd_ulcer.expanding(min_periods=1).max()

# 5. Máximo drawdown
print(f"Máximo drawdown: {((retornos_acumulados_mdd_ulcer - pico_mdd_ulcer) / pico_mdd_ulcer).min():.2%}")

# 6. Ulcer Index
# El drawdown para el Ulcer Index es (pico - valor_actual) / pico
_drawdown_valores_ulcer = (pico_mdd_ulcer - retornos_acumulados_mdd_ulcer) / pico_mdd_ulcer
print(f"Ulcer Index: {np.sqrt(np.sum(_drawdown_valores_ulcer**2) / len(rentabilidad_cartera)):.2f}")

# 7. VaR (Value at Risk) - Paramétrico Diario
_volatilidad_diaria_var = rentabilidad_cartera.std() # Calculamos la volatilidad diaria una vez para VaR
print(f"VaR 95% (Diario): {1.645 * _volatilidad_diaria_var:.2%}")
print(f"VaR 99% (Diario): {2.326 * _volatilidad_diaria_var:.2%}")
