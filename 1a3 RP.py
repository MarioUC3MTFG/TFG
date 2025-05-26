import yfinance as yf
import pandas as pd
import numpy as np
import riskfolio as rp
import matplotlib.pyplot as plt


# Rango de fechas
start = '2003-01-03'
end = '2021-12-30'

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
activos = yf.download(tickers, start = start, end = end, auto_adjust=False)
activos = activos.loc[:,('Adj Close', slice(None))]
activos.columns = tickers

# Compute daily returns and drop rows with missing values
returns = activos.pct_change().dropna(how="all")

def calculate_risk_parity_weights_rp(historical_returns):
    # Peso igualitario si menos de 2 observaciones
    if historical_returns.shape[0] < 2:
        n = historical_returns.shape[1]
        return np.ones(n) / n

    port = rp.Portfolio(returns=historical_returns)
    # Configuración para Risk Parity (basado en volatilidad)
    # rm='MV' utiliza la matriz de covarianza muestral.
    # Para 'gaussian' necesitarías más parámetros o una configuración específica.
    # 'vol' es la medida de riesgo estándar (volatilidad).
    port.assets_stats(method_mu='hist', method_cov='hist') # Usar retornos históricos para mu y cov

    weights_df = port.rp_optimization( 
        model='Classic', # Modelo clásico de paridad de riesgo (ERC)
        rm='MV',         # Medida de riesgo: Varianza (Mean-Variance)
        rf=0,            # Tasa libre de riesgo
        b=None           # Sin vector de contribución de riesgo específico (se busca igualdad)
    )

    if weights_df is None or weights_df.empty:
        n_assets = len(historical_returns.columns)
        return np.array([1/n_assets] * n_assets) # Fallback a pesos iguales

    return weights_df['weights'].values.flatten()

def backtest_risk_parity_rolling_window_rp(returns, lookback=756, rebalance_freq='ME'):
    portfolio_returns_series = pd.Series(index=returns.index[lookback:], dtype=float)
    all_weights = pd.DataFrame(index=returns.index[lookback:], columns=returns.columns, dtype=float)

    # Se encarga de que no haya rebalanceo hasta que haya pasado el período suficiente para calcular los datos por primera vez (3 años)
    rebalance_dates = returns.resample(rebalance_freq).first().index
    rebalance_dates = rebalance_dates[rebalance_dates >= returns.index[lookback]]

    # Necesario para la primera vez que se asignan pesos
    current_weights = None

    for i in range(lookback, len(returns.index)):
        current_date = returns.index[i]
        
        # Rebalanceo si es la fecha de rebalanceo y no es el primer cálculo de pesos
        if current_weights is None or current_date in rebalance_dates:
            # Seleccionar datos históricos para el cálculo de pesos (solo pasado)
            # `iloc` se usa para asegurar que tomamos `lookback` número de filas
            # `[:i]` asegura que solo usamos datos hasta el día *anterior* al `current_date` para evitar lookahead bias.
            historical_data_end_loc = returns.index.get_loc(current_date)
            historical_returns = returns.iloc[historical_data_end_loc - lookback : historical_data_end_loc]
            
            if historical_returns.shape[0] < lookback: # Asegurarse que tenemos suficientes datos
                # Si al inicio del backtest no hay suficientes datos con `[:i]`,
                # podríamos decidir esperar o usar los datos disponibles.
                # Por simplicidad, si esto ocurre al principio, lo saltamos o usamos pesos iguales.
                # Sin embargo, el bucle for ya empieza en `lookback`, así que esto
                # se maneja por la selección de `historical_returns`
                if current_weights is None: # Para el primer cálculo si no hay suficientes datos
                    n_assets = len(returns.columns)
                    current_weights = np.array([1/n_assets] * n_assets)
                # Si no, mantenemos los pesos anteriores hasta tener suficientes datos.
            else:
                current_weights = calculate_risk_parity_weights_rp(historical_returns)
        
        # Calcular el retorno del portafolio para el día actual con los pesos calculados
        period_return = np.sum(current_weights * returns.loc[current_date])
        portfolio_returns_series[current_date] = period_return
        all_weights.loc[current_date] = current_weights
        
    return portfolio_returns_series.dropna(), all_weights.dropna()

# Ejecutar el backtest
# lookback: número de días para calcular la covarianza y los pesos (aprox. 1 año)
# rebalance_freq: 'M' para mensual, 'Q' para trimestral, 'A' para anual
lookback_period = 756
rebalance_frequency = 'ME' # Rebalanceo mensual

rp_portfolio_returns, rp_weights = backtest_risk_parity_rolling_window_rp(
    returns,
    lookback=lookback_period,
    rebalance_freq=rebalance_frequency
)

print("\nRetornos del portafolio Risk Parity (primeras filas):")
print(rp_portfolio_returns.head())
print("\nPesos del portafolio Risk Parity (últimas filas para ver los más recientes):")
print(rp_weights.tail())

def calculate_performance_metrics(returns_series):
    """
    Calcula métricas de rendimiento para una serie de retornos.
    """
    annual_return = returns_series.mean() * 252  # Asumiendo 252 días de trading al año
    volatility = returns_series.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    return pd.Series({
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    })

# Calcular métricas para el portafolio Risk Parity
rp_metrics = calculate_performance_metrics(rp_portfolio_returns)
print("\nMétricas de Rendimiento del Portafolio Risk Parity:")
print(rp_metrics)

# Crear un portafolio de igual peso (Equal Weight) para comparación
n_assets = returns.shape[1]
equal_weights = np.array([1/n_assets] * n_assets)
# Alineamos índices temporales usando .loc en lugar de []
ew_portfolio_returns = returns.loc[rp_portfolio_returns.index].dot(equal_weights)

ew_metrics = calculate_performance_metrics(ew_portfolio_returns)
print("\nMétricas de Rendimiento del Portafolio Equal Weight:")
print(ew_metrics)


# # Graficar los retornos acumulados
# def plot_strategy_comparison(returns_dict):
#     cumulative_returns_df = pd.DataFrame()
#     for strategy, returns_series in returns_dict.items():
#         if not returns_series.empty:
#             cumulative_returns_df[strategy] = (1 + returns_series).cumprod()
    

#     cumulative_returns_df.plot(figsize=(12, 7))
#     plt.title('Retornos Acumulados de las Estrategias')
#     plt.ylabel('Valor del Portafolio (normalizado a 1)')
#     plt.xlabel('Fecha')
#     plt.grid(True)
#     plt.show()


# # Comparar Risk Parity con Equal Weight
# strategies_to_plot = {
#     'Risk Parity (riskfolio-lib)': rp_portfolio_returns,
#     'Equal Weight': ew_portfolio_returns
# }

# plot_strategy_comparison(strategies_to_plot)

# # Graficar la evolución de los pesos del portafolio Risk Parity
# rp_weights.plot(figsize=(12, 7))
# plt.title('Evolución de los Pesos del Portafolio Risk Parity')
# plt.ylabel('Peso del Activo')
# plt.xlabel('Fecha')
# plt.legend(title='Activos')
# plt.grid(True)
# plt.show()
