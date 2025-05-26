import yfinance as yf
from pathlib import Path

start = '2003-01-03'
end = '2021-12-30'

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

data = yf.download(tickers, start = start, end = end, auto_adjust=False)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = tickers

destino = Path("data.csv")
data.to_csv(destino, sep=",", decimal=".", float_format="%.6f")
print(f"✅ CSV limpio guardado en {destino.resolve()}")
