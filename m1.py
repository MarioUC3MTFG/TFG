import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---------- 1) Leer y limpiar datos ----------
precios = pd.read_csv(
    "precios.csv",
    sep=",",           # separador por coma
    decimal=".",       # decimal por punto
    index_col=0
)

# Asegurar que el índice se interpreta como fechas
precios.index = pd.to_datetime(precios.index, format="%Y-%m-%d")
precios = precios.sort_index()

rend_d = precios.pct_change(fill_method=None)
rend_d = rend_d.where(rend_d.abs() < 0.3).dropna(how="all")   # quita saltos ±30 %

# ---------- 2) Parámetros del back-test ----------
ventana = 36        # meses de histórico para estimar μ y Σ
rf      = 0.02      # 2 % anual libre de riesgo
shorts  = False     # no se permiten cortos

# ---------- 3) Función Sharpe máximo ----------
def tangency(mu, Sigma, rf, shorts):
    n = len(mu)
    def neg_sharpe(w):
        r, s = w @ mu - rf, np.sqrt(w @ Sigma @ w)
        return -r / s
    cons   = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = None if shorts else [(0, None)] * n
    res = minimize(neg_sharpe, np.repeat(1/n, n),
                   constraints=cons, bounds=bounds, method="SLSQP")
    return res.x

# ---------- 4) Agregar rendimientos a nivel mensual ----------
rend_m = rend_d.resample("ME").apply(lambda x: (1 + x).prod() - 1)  # "ME" = fin de mes
months  = rend_m.index

# ---------- 5) Bucles rolling ----------
ret_mkt, ret_eq = [], []

for i in range(ventana, len(months)):
    fin_est = months[i - 1]              # último mes de la ventana
    datos   = rend_d.loc[months[i-ventana]:fin_est]   # 36 m previos (diario)

    mu  = datos.mean() * 252
    Sig = datos.cov()  * 252
    mu, Sig = mu.dropna(), Sig.loc[mu.index, mu.index]   # sincroniza

    if len(mu) < 2:                       # por seguridad
        ret_mkt.append(np.nan)
        ret_eq.append(np.nan)
        continue

    w_mkt = tangency(mu.values, Sig.values, rf, shorts)

    mes_act = rend_m.iloc[i][mu.index]    # retorno de cada activo ese mes
    ret_mkt.append(np.nansum(w_mkt * mes_act))

    ret_eq.append(mes_act.mean())         # equiponderada (rebalanceada)

# ---------- 6) Convertir a Series alineadas ----------
bt_months = months[ventana:]
ret_mkt   = pd.Series(ret_mkt, index=bt_months, name="MKT")
ret_eq    = pd.Series(ret_eq,  index=bt_months, name="EQUI")

# ---------- 7) Métricas de performance ----------
def anualiza(r):
    total = (1 + r).prod()
    años  = len(r) / 12
    return total**(1/años) - 1

def vol(r): return r.std() * np.sqrt(12)

for nombre, serie in [("Equiponderada", ret_eq), ("Cartera MKT", ret_mkt)]:
    print(f"\n=== {nombre} ===")
    print(f"Rentabilidad compuesta anual : {anualiza(serie):.2%}")
    print(f"Volatilidad anual           : {vol(serie):.2%}")
    sharpe = (anualiza(serie) - rf) / vol(serie)
    print(f"Sharpe (rf={rf:.0%})         : {sharpe:.2f}")

# ---------- 8) Curvas de capital ----------
capital = pd.DataFrame({
    "Equiponderada": (1 + ret_eq).cumprod(),
    "Cartera MKT":   (1 + ret_mkt).cumprod()
})

capital.plot(figsize=(10,5), title="Evolución del capital (back-test rolling)")
plt.ylabel("Multiplicador del capital inicial")
plt.grid(True)
plt.tight_layout()
plt.show()
