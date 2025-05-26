import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.4%}'.format

START_DATE_DOWNLOAD = '2003-01-03' 
END_DATE_DOWNLOAD = '2021-12-30'   

ASSETS = [    "XOM","GE","MSFT","C","BAC","WMT","PG","PFE","JNJ","AIG",
            "IBM","CVX","INTC","WFC","T","KO","VZ","PEP","HPQ","HD","AMGN",
            "UPS","COP","QCOM","MRK","UNH","ORCL","ABT","AXP","JPM","MO",
            "GS","MS","LLY","MDT","BA","CMCSA","MMM","SLB","CSCO","CL",
            "FNMA","VLO","CAT","TGT","DE","OXY","DD","FDX","LOW","CVS",
            "PRU","HAL","EXC","STT","COF","LMT","ALL","KR","DUK","MET",
            "COST","EMR","SBUX","CI","PNC","PGR","ITW","TXT","SO","HON",
            "PPG","BSX","MMC","NOC","EOG","DVN","NUE","FCX","USB","NSC",
            "UNP","KEY","BAX","AZO","GPC","DHR","BEN","TMO","TROW","BXP",
            "SPG","KMB","SCHW","RF","WMB","PAYX", "D", "GOLD", "TLT"] #

TRADING_DAYS_PER_YEAR = 252
LOOKBACK_YEARS = 3  = LOOKBACK_YEARS * TRADING_DAYS_PER_YEAR
REBALANCE_FREQUENCY = 'ME'  

HRP_MODEL_PARAMS = {
    'model': 'HRP',
    'codependence': 'pearson',
    'rm': 'MV', 
    'rf': 0,    
    'linkage': 'average',
    'max_k': 10,
    'leaf_order': True
}

print(f"Descargando datos para {len(ASSETS)} tickers...")
asset_data_downloaded = yf.download(ASSETS,
                                    start=START_DATE_DOWNLOAD,
                                    end=END_DATE_DOWNLOAD,
                                    auto_adjust=False,
                                    progress=True)

asset_prices = asset_data_downloaded.loc[:, ('Adj Close', slice(None))]
asset_prices.columns = ASSETS
asset_prices.dropna(axis=1, how='all', inplace=True)
current_assets = asset_prices.columns.tolist()
ASSETS = current_assets 
print(f"Datos descargados para {len(ASSETS)} activos.")

asset_returns = asset_prices.pct_change().dropna() #

if len(asset_returns) < LOOKBACK_DAYS:
    print(f"Datos de retornos insuficientes ({len(asset_returns)} filas) para el lookback de {LOOKBACK_DAYS} días.")
    exit()

print(f"Cálculo de retornos diarios completado. {len(asset_returns)} observaciones.")

#  FUNCIONES PARA EL MODELO Y BACKTESTING

def calculate_hrp_weights_riskfolio(historical_returns: pd.DataFrame,
                                    model_params: dict) -> np.ndarray:
    """
    Calcula los pesos óptimos del portafolio HRP usando Riskfolio-Lib.
    Simplificado para seguir el "modelo" y reducir comprobaciones.
    """
    num_assets = historical_returns.shape[1]
    port = rp.HCPortfolio(returns=historical_returns)

    weights_df = port.optimization(
        model=model_params['model'],
        codependence=model_params['codependence'],
        rm=model_params['rm'],
        rf=model_params['rf'],
        linkage=model_params['linkage'],
        max_k=model_params['max_k'],
        leaf_order=model_params['leaf_order']
    )

    if weights_df is None or weights_df.empty:
     
        return np.ones(num_assets) / num_assets
    
    return weights_df['weights'].values.flatten()


def run_backtest_rolling_window(
    all_returns: pd.DataFrame,
    lookback_period_days: int,
    rebalance_rule: str,
    weight_calculation_func: callable,
    model_parameters: dict
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Realiza un backtest con ventana móvil. Simplificado.
    """
    print(f"\nIniciando backtest: Lookback {lookback_period_days} días, Rebalanceo {rebalance_rule}.")

    portfolio_result_index = all_returns.index[lookback_period_days:]
    portfolio_returns_series = pd.Series(index=portfolio_result_index, dtype=float, name="Portfolio_Returns")
    all_weights_df = pd.DataFrame(index=portfolio_result_index, columns=all_returns.columns, dtype=float)

    rebalance_trigger_dates = all_returns.resample(rebalance_rule).first().index
    rebalance_trigger_dates = rebalance_trigger_dates[rebalance_trigger_dates >= portfolio_result_index[0]]

    current_weights = None
    
    for i in range(lookback_period_days, len(all_returns.index)):
        current_date = all_returns.index[i]

        if current_weights is None or current_date in rebalance_trigger_dates:
            loc_current_date = all_returns.index.get_loc(current_date)
            historical_returns_window = all_returns.iloc[loc_current_date - lookback_period_days : loc_current_date]
            
            if historical_returns_window.shape[0] < lookback_period_days // 2 :
                 if current_weights is None: 
                    num_assets_fallback = all_returns.shape[1]
                    current_weights = np.array([1/num_assets_fallback] * num_assets_fallback)

            else:

                current_weights = weight_calculation_func(historical_returns_window, model_parameters)
        
        if current_weights is not None:
            period_return = np.sum(current_weights * all_returns.loc[current_date])
            portfolio_returns_series.loc[current_date] = period_return
            all_weights_df.loc[current_date] = current_weights
        else: 
            portfolio_returns_series.loc[current_date] = np.nan
            
    final_portfolio_returns = portfolio_returns_series.dropna()
    final_weights_df = all_weights_df.loc[final_portfolio_returns.index].dropna(how='all')

    print("Backtest finalizado.")
    return final_portfolio_returns, final_weights_df

# FUNCIONES DE MÉTRICAS Y GRÁFICOS

def calculate_performance_metrics(returns_series: pd.Series,
                                  trading_days_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    if returns_series.empty or returns_series.isnull().all():
        return pd.Series({
            'Retorno Anualizado': 0, 'Volatilidad Anualizada': 0, 'Ratio de Sharpe': 0,
            'Máximo Drawdown': 0, 'Retorno Acumulado Total': 1.0
        }, dtype=float)

    annual_return = returns_series.mean() * trading_days_year
    volatility = returns_series.std() * np.sqrt(trading_days_year)
    sharpe_ratio = annual_return / volatility if volatility != 0 else 0
    
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    total_cumulative_return = cumulative_returns.iloc[-1] if not cumulative_returns.empty else 1.0

    metrics = pd.Series({
        'Retorno Anualizado': annual_return,
        'Volatilidad Anualizada': volatility,
        'Ratio de Sharpe': sharpe_ratio,
        'Máximo Drawdown': max_drawdown,
        'Retorno Acumulado Total': total_cumulative_return
    })
    return metrics

def plot_cumulative_returns(returns_dict: dict, title: str = 'Retornos Acumulados de Estrategias'):
    cumulative_returns_df = pd.DataFrame()
    for strategy_name, returns_s in returns_dict.items():
        if not returns_s.empty:
            cumulative_returns_df[strategy_name] = (1 + returns_s).cumprod()

    if not cumulative_returns_df.empty:
        cumulative_returns_df.plot(figsize=(12, 7))
        plt.title(title)
        plt.ylabel('Valor del Portafolio (Normalizado a 1)')
        plt.xlabel('Fecha')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("No hay datos de retornos acumulados para graficar.")

def plot_asset_weights_evolution(weights_df: pd.DataFrame, strategy_name: str):
    if not weights_df.empty:
        if weights_df.shape[1] > 20: 
            plt.figure(figsize=(14, 8))
            plt.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns)
            plt.title(f'Evolución de Pesos de Activos: {strategy_name}')
            plt.ylabel('Peso del Activo')
            plt.xlabel('Fecha')
            plt.figlegend(labels=weights_df.columns, loc='center left', bbox_to_anchor=(0.9, 0.5), ncol=max(1, len(weights_df.columns)//30))
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
        else:
            weights_df.plot(figsize=(14, 8), kind='area', stacked=True)
            plt.title(f'Evolución de Pesos de Activos: {strategy_name}')
            plt.ylabel('Peso del Activo')
            plt.xlabel('Fecha')
            plt.legend(title='Activos', loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.show()
    else:
        print(f"No hay datos de pesos para graficar para la estrategia {strategy_name}.")

#  EJECUCIÓN DEL BACKTEST Y ANÁLISIS DE RESULTADOS

hrp_portfolio_returns, hrp_asset_weights = run_backtest_rolling_window(
    all_returns=asset_returns,
    lookback_period_days=LOOKBACK_DAYS,
    rebalance_rule=REBALANCE_FREQUENCY,
    weight_calculation_func=calculate_hrp_weights_riskfolio,
    model_parameters=HRP_MODEL_PARAMS
)

if not hrp_portfolio_returns.empty:
    print("\n--- Resultados del Portafolio HRP ---")
    hrp_performance_metrics = calculate_performance_metrics(hrp_portfolio_returns)
    print("\nMétricas de Rendimiento del Portafolio HRP:")
    print(hrp_performance_metrics)

   
    aligned_asset_returns_for_ew = asset_returns.loc[hrp_portfolio_returns.index]
    
    if not aligned_asset_returns_for_ew.empty:
        num_assets_ew = aligned_asset_returns_for_ew.shape[1]
        equal_weights_values = np.array([1/num_assets_ew] * num_assets_ew)
        ew_portfolio_returns = aligned_asset_returns_for_ew.dot(equal_weights_values)
        ew_portfolio_returns.name = "EqualWeight_Returns"

        ew_performance_metrics = calculate_performance_metrics(ew_portfolio_returns)
        print("\nMétricas de Rendimiento del Portafolio Equal Weight (EW):")
        print(ew_performance_metrics)

        plot_cumulative_returns({
            f'HRP ({LOOKBACK_YEARS}a, {REBALANCE_FREQUENCY})': hrp_portfolio_returns,
            'Equal Weight': ew_portfolio_returns
        })
    else:
        plot_cumulative_returns({
            f'HRP ({LOOKBACK_YEARS}a, {REBALANCE_FREQUENCY})': hrp_portfolio_returns
        })
        
    plot_asset_weights_evolution(hrp_asset_weights, f'HRP ({LOOKBACK_YEARS}a, {REBALANCE_FREQUENCY})')
else:
    print("\nEl backtesting HRP no generó resultados.")

print("\nAnálisis finalizado.")
