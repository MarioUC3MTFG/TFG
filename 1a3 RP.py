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
    port.assets_stats(method_mu='hist', method_cov='hist') 

    weights_df = port.rp_optimization( 
        model='Classic', 
        rm='MV',         
        rf=0,            
        b=None           
    )

    if weights_df is None or weights_df.empty:
        n_assets = len(historical_returns.columns)
        return np.array([1/n_assets] * n_assets) 

    return weights_df['weights'].values.flatten()

def backtest_risk_parity_rolling_window_rp(returns, lookback=756, rebalance_freq='ME'):
    portfolio_returns_series = pd.Series(index=returns.index[lookback:], dtype=float)
    all_weights = pd.DataFrame(index=returns.index[lookback:], columns=returns.columns, dtype=float)

    
    rebalance_dates = returns.resample(rebalance_freq).first().index
    rebalance_dates = rebalance_dates[rebalance_dates >= returns.index[lookback]]

    
    current_weights = None

    for i in range(lookback, len(returns.index)):
        current_date = returns.index[i]
        
      
        if current_weights is None or current_date in rebalance_dates:
            historical_data_end_loc = returns.index.get_loc(current_date)
            historical_returns = returns.iloc[historical_data_end_loc - lookback : historical_data_end_loc]
            
            if historical_returns.shape[0] < lookback: 
                if current_weights is None: 
                    n_assets = len(returns.columns)
                    current_weights = np.array([1/n_assets] * n_assets)
    
            else:
                current_weights = calculate_risk_parity_weights_rp(historical_returns)
        
  
        period_return = np.sum(current_weights * returns.loc[current_date])
        portfolio_returns_series[current_date] = period_return
        all_weights.loc[current_date] = current_weights
        
    return portfolio_returns_series.dropna(), all_weights.dropna()

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
    annual_return = returns_series.mean() * 252  
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


rp_metrics = calculate_performance_metrics(rp_portfolio_returns)
print("\nMétricas de Rendimiento del Portafolio Risk Parity:")
print(rp_metrics)

n_assets = returns.shape[1]
equal_weights = np.array([1/n_assets] * n_assets)
ew_portfolio_returns = returns.loc[rp_portfolio_returns.index].dot(equal_weights)

ew_metrics = calculate_performance_metrics(ew_portfolio_returns)
print("\nMétricas de Rendimiento del Portafolio Equal Weight:")
print(ew_metrics)


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
