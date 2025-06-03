import yfinance as yf
import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt

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

# Calcular retornos diarios
retornos = activos.pct_change().dropna()

# Primero definimos la función de Risk parity que posteriormente llamaremos en cada rebalanceo del bucle. 
def pesos_RP(retornos):
    port = rp.Portfolio(returns = retornos)
    port.assets_stats(method_mu = 'hist', method_cov = 'hist')
    pesos = port.rp_optimization(model = 'Classic', rm = 'MV', rf = 0)
    # Elegimos la columan de pesos y la transformamos en un array
    return pesos['weights']

# Calculamos el índice de fechas del último día de cada mes (Evitamos repetir la tarea en el bucle) 
rebalance_freq='ME'
fechas_rebalanceo = retornos.resample(rebalance_freq).mean().index

def backtest_RP(retornos, lookback = 756):

    #Inicializamos las variables
    rentabilidad_cartera = []
    ponderaciones = None

    for date in retornos.index[lookback:]:
        if ponderaciones is None or date in fechas_rebalanceo:
            Historico = retornos.loc[:date].iloc[-lookback:]
            ponderaciones = pesos_RP(Historico)      

        # Calculo de retornos
        retornos_periodo = np.sum(ponderaciones * retornos.loc[date], axis = 0)
        rentabilidad_cartera.append(retornos_periodo)


    return pd.Series(rentabilidad_cartera, index=retornos.index[lookback:])

# Aquí extraigo el valor de los retornos de la función y se la doy a la nueva variable creada  
rentabilidad_cartera = backtest_RP(retornos)


print("Métricas de la cartera de Risk Parity")
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

