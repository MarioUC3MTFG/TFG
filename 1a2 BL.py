# ---------------------------
# 📦 BLACK-LITTERMAN COMPLETO Y ROBUSTO — DELTA JUSTIFICADO
# ---------------------------

import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import risk_models, expected_returns
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

# ---------------------------
# 1. Cargar precios de activos
# ---------------------------
try:
    prices = pd.read_csv("precios.csv", index_col=0, parse_dates=True)
    prices = prices.sort_index()
    print("✅ precios.csv cargado correctamente.")
except Exception as e:
    print("❌ Error al cargar precios.csv:", e)
    raise

# ---------------------------
# 2. Descargar precios del índice SPY
# ---------------------------
try:
    spy_data = yf.download("SPY", start="2003-01-01", end="2021-12-31", auto_adjust=False)
    market_prices = spy_data["Adj Close"]
    print("✅ Precios de SPY descargados correctamente.")
except Exception as e:
    print("❌ Error al descargar SPY:", e)
    raise

# ---------------------------
# 3. Calcular delta basado en rentabilidad esperada y varianza del mercado
# ---------------------------
expected_market_return = 0.10  # Supuesto: 10 % anual
returns_spy = market_prices.pct_change().dropna()
volatility = returns_spy.std() * np.sqrt(252)
variance = volatility**2
delta = (expected_market_return / variance).item()
print(f"✅ Delta calculado con rentabilidad esperada del 10 %: {round(delta, 4)}")

# ---------------------------
# 4. Configuración de rebalanceo
# ---------------------------
window_months = 36
rebalance_dates = prices.resample("ME").last().index[window_months:]

view_periods = {
    (2006, 2011): ["XOM", "CVX", "COP", "VLO", "OXY", "SLB", "HAL"],
    (2011, 2016): ["MSFT", "CSCO", "IBM", "INTC", "QCOM"],
    (2016, 2021): ["PFE", "JNJ", "MRK", "UNH", "ABT", "WYE", "LLY"],
}

# ---------------------------
# 5. Bucle de rebalanceo mensual
# ---------------------------
rebalance_results = []

for date in rebalance_dates:
    start = date - pd.DateOffset(months=window_months)
    precios_ventana = prices.loc[start:date].dropna(axis=1)

    if precios_ventana.shape[1] < 2:
        continue

    # (1) Covarianza
    cov_matrix = risk_models.sample_cov(precios_ventana)

    # (2) Prior: retornos históricos
    pi = expected_returns.mean_historical_return(precios_ventana)

    # (3) Determinar views activas
    year = date.year
    view_tickers = []
    for (y1, y2), tickers in view_periods.items():
        if y1 <= year < y2:
            view_tickers = [t for t in tickers if t in precios_ventana.columns]
            break

    if not view_tickers:
        continue

    absolute_views = {ticker: 0.05 for ticker in view_tickers}
    view_confidences = [0.6] * len(absolute_views)

    # (4) Modelo Black-Litterman
    bl = BlackLittermanModel(
        cov_matrix,
        pi=pi,
        absolute_views=absolute_views,
        view_confidences=view_confidences,
        omega="idzorek",
        risk_aversion=delta
    )

    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # (5) Optimización con restricción del 10 % por activo
    ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(0, 0.10))
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # (6) Guardar resultados
    rebalance_results.append({
        "date": date,
        "weights": cleaned_weights
    })

# ---------------------------
# 6. Exportar resultados
# ---------------------------
df_resultados = pd.DataFrame(rebalance_results)
df_resultados.to_pickle("pesos_blacklitterman.pkl")
print("✅ Resultados guardados en 'pesos_blacklitterman.pkl'")

# ---------------------------
# 7. Evaluar la estrategia Black-Litterman
# ---------------------------
from collections import defaultdict
import matplotlib.pyplot as plt

returns = prices.pct_change().dropna()
cartera = pd.Series(data=np.nan, index=returns.index, dtype="float64")
if not cartera.empty:
    cartera.iloc[0] = 1
else:
    raise ValueError("❌ La serie de retornos está vacía. Revisa 'prices.csv' o fechas.")

pesos_df = pd.DataFrame([
    pd.Series(w["weights"], name=w["date"])
    for w in rebalance_results
])

pesos_df = pesos_df.sort_index().reindex(returns.index, method="ffill").fillna(0)
pesos_df = pesos_df.loc[:, pesos_df.columns.intersection(returns.columns)]
ret_cartera = (pesos_df * returns[pesos_df.columns]).sum(axis=1)
valor = (1 + ret_cartera).cumprod()

# --- Estadísticas clave
cagr = (valor.iloc[-1]) ** (252 / len(valor)) - 1
vol = ret_cartera.std() * np.sqrt(252)
sharpe = (cagr - 0.02) / vol  # rf = 2 %

# --- Mostrar resultados
print("\n=== Cartera Black-Litterman ===")
print(f"Rentabilidad compuesta anual : {cagr * 100:.2f}%")
print(f"Volatilidad anual           : {vol * 100:.2f}%")
print(f"Sharpe (rf=2%)              : {sharpe:.2f}")

valor.plot(title="Evolución del valor de la cartera BL", figsize=(10, 5))
plt.ylabel("Valor acumulado")
plt.xlabel("Fecha")
plt.grid(True)
plt.tight_layout()
plt.show()

pesos_promedio = pesos_df.groupby(pesos_df.index.year).mean()
print("\nPesos medios por año:")
print(pesos_promedio.round(3).head())

hhi = (pesos_df ** 2).sum(axis=1).mean()
print(f"\nÍndice de concentración (HHI): {hhi:.4f}")
