"""Tarik harga harian IHSG, emas, USD/IDR dan saham watchlist dari Yahoo Finance (via yfinance).
Jalan di GitHub Actions. Riwayat disimpan di data/<simbol>.csv (hanya hari bursa yang sudah selesai).
Snapshot terbaru (boleh belum final) disimpan di data/_snapshot.csv dengan waktu tarik."""
import os, sys, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

WIB = ZoneInfo("Asia/Jakarta")
IDX = ("BBCA BBRI BMRI BBNI TLKM ASII UNTR ITMG PTBA ADMF BFIN JPFA PGEO AMMN MTEL ISAT "
       "KLBF ICBP INDF MYOR IKBI BREN DSSA BYAN DCII AGII IFSH RLCO ROCK SOTS WBSA "
       "STRK TFCO DAAZ ARKO").split()
SYMBOLS = {"^JKSE": "IHSG", "GC=F": "EMAS_USD_OZ", "IDR=X": "USDIDR"}
SYMBOLS.update({f"{t}.JK": t for t in IDX})
os.makedirs("data", exist_ok=True)

now = dt.datetime.now(WIB)
# bar hari ini dianggap final hanya setelah 16:30 WIB (bursa tutup 16:00, pre-closing 16:10)
today_final = now.time() >= dt.time(16, 30)
cutoff = now.date() if today_final else now.date() - dt.timedelta(days=1)

snap_rows, failed = [], []
for sym, name in SYMBOLS.items():
    path = f"data/{name}.csv"
    period = "10d" if os.path.exists(path) else "max"
    try:
        df = yf.Ticker(sym).history(period=period, interval="1d", auto_adjust=False)
        if df.empty:
            raise ValueError("data kosong")
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        last = df.iloc[-1]
        snap_rows.append({"simbol": name, "yahoo": sym, "tanggal_bar": last["Date"],
                          "close": last["Close"], "volume": last["Volume"],
                          "ditarik_wib": now.strftime("%Y-%m-%d %H:%M"),
                          "final": bool(last["Date"] <= cutoff)})
        hist = df[df["Date"] <= cutoff]
        if os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["Date"])
            old["Date"] = old["Date"].dt.date
            hist = pd.concat([old, hist]).drop_duplicates("Date", keep="last")
        hist.sort_values("Date").to_csv(path, index=False)
    except Exception as e:  # satu simbol gagal tidak menghentikan yang lain
        failed.append(f"{sym}: {e}")

pd.DataFrame(snap_rows).to_csv("data/_snapshot.csv", index=False)
with open("data/_status.txt", "w") as f:
    f.write(f"ditarik_wib={now:%Y-%m-%d %H:%M}\nberhasil={len(snap_rows)}\ngagal={len(failed)}\n")
    f.write("\n".join(failed) + ("\n" if failed else ""))
print(open("data/_status.txt").read())
sys.exit(0 if snap_rows else 1)
