"""Hitung indikator teknikal, pola candlestick, dan pola grafik dari data/<simbol>.csv.
Keluaran: data/_indikator.csv (satu baris per simbol). Semua hitungan bersifat heuristik untuk belajar,
bukan prediksi. Butuh kolom Date, Open, High, Low, Close, Volume."""
import glob, os, sys
import numpy as np
import pandas as pd

# ---------- Rata-rata bergerak ----------
def sma(s, n): return s.rolling(n).mean()
def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)
def ema(s, n): return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

# ---------- Candlestick ----------
def _parts(r):
    o, h, l, c = r["Open"], r["High"], r["Low"], r["Close"]
    body, rng = abs(c - o), max(h - l, 1e-9)
    return dict(o=o, h=h, l=l, c=c, body=body, rng=rng,
                up=h - max(o, c), lo=min(o, c) - l, green=c > o, red=c < o)

def _prior_trend(df, end, look=5, thr=0.01):
    """Tren sebelum bar ke-end: +1 naik, -1 turun, 0 datar (berdasarkan close)."""
    if end - look < 0: return 0
    a, b = df["Close"].iloc[end - look], df["Close"].iloc[end - 1]
    ch = b / a - 1 if a else 0
    return 1 if ch > thr else -1 if ch < -thr else 0

def candle_patterns(df):
    """Pola pada bar terakhir (butuh minimal 6 bar). Mengembalikan daftar (nama, arah, jenis)."""
    out, n = [], len(df)
    if n < 6: return out
    p = [_parts(df.iloc[i]) for i in range(n - 5, n)]   # p[-1] = bar terakhir
    t = _prior_trend(df, n - 1)
    c0, c1, c2 = p[-1], p[-2], p[-3]
    big = lambda x: x["body"] >= 0.6 * x["rng"]
    # Satu bar
    if c0["body"] <= 0.1 * c0["rng"]:
        out.append(("Doji", "netral", "keraguan"))
    if c0["body"] >= 0.95 * c0["rng"]:
        out.append(("Marubozu hijau" if c0["green"] else "Marubozu merah",
                    "naik" if c0["green"] else "turun", "kelanjutan"))
    if c0["lo"] >= 2 * c0["body"] and c0["up"] <= 0.3 * max(c0["body"], 1e-9) + 0.05 * c0["rng"] and c0["body"] > 0:
        if t < 0: out.append(("Hammer", "naik", "pembalikan"))
        elif t > 0: out.append(("Hanging man", "turun", "pembalikan"))
    if c0["up"] >= 2 * c0["body"] and c0["lo"] <= 0.3 * max(c0["body"], 1e-9) + 0.05 * c0["rng"] and c0["body"] > 0:
        if t > 0: out.append(("Shooting star", "turun", "pembalikan"))
        elif t < 0: out.append(("Inverted hammer", "naik", "pembalikan"))
    # Dua bar
    if c1["red"] and c0["green"] and c0["c"] >= c1["o"] and c0["o"] <= c1["c"] and c0["body"] > c1["body"] and t <= 0:
        out.append(("Bullish engulfing", "naik", "pembalikan"))
    if c1["green"] and c0["red"] and c0["o"] >= c1["c"] and c0["c"] <= c1["o"] and c0["body"] > c1["body"] and t >= 0:
        out.append(("Bearish engulfing", "turun", "pembalikan"))
    if big(c1) and c0["body"] < c1["body"] and max(c0["o"], c0["c"]) <= max(c1["o"], c1["c"]) and min(c0["o"], c0["c"]) >= min(c1["o"], c1["c"]):
        if c1["red"] and c0["green"] and t < 0: out.append(("Bullish harami", "naik", "pembalikan"))
        if c1["green"] and c0["red"] and t > 0: out.append(("Bearish harami", "turun", "pembalikan"))
    mid1 = (c1["o"] + c1["c"]) / 2
    if c1["red"] and big(c1) and c0["green"] and c0["o"] < c1["c"] and mid1 < c0["c"] < c1["o"] and t < 0:
        out.append(("Piercing line", "naik", "pembalikan"))
    if c1["green"] and big(c1) and c0["red"] and c0["o"] > c1["c"] and c1["o"] < c0["c"] < mid1 and t > 0:
        out.append(("Dark cloud cover", "turun", "pembalikan"))
    # Tiga bar
    if c2["red"] and big(c2) and c1["body"] <= 0.35 * c1["rng"] + 1e-9 and c0["green"] and big(c0) and c0["c"] > (c2["o"] + c2["c"]) / 2 and t < 0:
        out.append(("Morning star", "naik", "pembalikan"))
    if c2["green"] and big(c2) and c1["body"] <= 0.35 * c1["rng"] + 1e-9 and c0["red"] and big(c0) and c0["c"] < (c2["o"] + c2["c"]) / 2 and t > 0:
        out.append(("Evening star", "turun", "pembalikan"))
    if all(x["green"] and big(x) for x in (c2, c1, c0)) and c2["c"] < c1["c"] < c0["c"] and c1["o"] > c2["o"] and c0["o"] > c1["o"]:
        out.append(("Three white soldiers", "naik", "pembalikan" if t < 0 else "kelanjutan"))
    if all(x["red"] and big(x) for x in (c2, c1, c0)) and c2["c"] > c1["c"] > c0["c"] and c1["o"] < c2["o"] and c0["o"] < c1["o"]:
        out.append(("Three black crows", "turun", "pembalikan" if t > 0 else "kelanjutan"))
    # Lima bar: rising / falling three methods (kelanjutan)
    a, m1, m2, m3, z = p[-5], p[-4], p[-3], p[-2], p[-1]
    if a["green"] and big(a) and z["green"] and big(z) and all(x["red"] and x["body"] < a["body"] for x in (m1, m2, m3)) \
            and all(x["l"] > a["l"] and x["h"] < a["h"] + a["rng"] * 0.1 for x in (m1, m2, m3)) and z["c"] > a["c"]:
        out.append(("Rising three methods", "naik", "kelanjutan"))
    if a["red"] and big(a) and z["red"] and big(z) and all(x["green"] and x["body"] < a["body"] for x in (m1, m2, m3)) \
            and all(x["h"] < a["h"] and x["l"] > a["l"] - a["rng"] * 0.1 for x in (m1, m2, m3)) and z["c"] < a["c"]:
        out.append(("Falling three methods", "turun", "kelanjutan"))
    return out

# ---------- Pola grafik (berbasis pivot) ----------
def pivots(df, k=4):
    """Pivot high/low: bar yang paling tinggi/rendah di antara k bar kiri dan kanan."""
    hi, lo = [], []
    H, L = df["High"].values, df["Low"].values
    for i in range(k, len(df) - k):
        if H[i] == H[i - k:i + k + 1].max(): hi.append(i)
        if L[i] == L[i - k:i + k + 1].min(): lo.append(i)
    def merge(idx, arr, pick):
        out = []
        for i in idx:  # pivot yang berdekatan (nilai sama) digabung, ambil yang paling ekstrem
            if out and i - out[-1] <= k:
                if pick(arr[i], arr[out[-1]]): out[-1] = i
            else:
                out.append(i)
        return out
    return merge(hi, H, lambda a, b: a > b), merge(lo, L, lambda a, b: a < b)

def chart_patterns(df, lookback=140):
    out = []
    d = df.tail(lookback).reset_index(drop=True)
    if len(d) < 40: return out
    hi, lo = pivots(d)
    H, L, C, V = d["High"].values, d["Low"].values, d["Close"].values, d["Volume"].values
    last = C[-1]
    # Double top / bottom
    if len(hi) >= 2:
        i, j = hi[-2], hi[-1]
        if j - i >= 10 and abs(H[i] / H[j] - 1) <= 0.03:
            neck = L[i:j + 1].min()
            if neck <= min(H[i], H[j]) * 0.97 and j >= len(d) - 25:
                out.append(("Double top terkonfirmasi" if last < neck else "Double top potensial", "turun", "pembalikan"))
    if len(lo) >= 2:
        i, j = lo[-2], lo[-1]
        if j - i >= 10 and abs(L[i] / L[j] - 1) <= 0.03:
            neck = H[i:j + 1].max()
            if neck >= max(L[i], L[j]) * 1.03 and j >= len(d) - 25:
                out.append(("Double bottom terkonfirmasi" if last > neck else "Double bottom potensial", "naik", "pembalikan"))
    # Head and shoulders / inverse
    if len(hi) >= 3:
        a, b, c = hi[-3], hi[-2], hi[-1]
        if H[b] > H[a] and H[b] > H[c] and abs(H[a] / H[c] - 1) <= 0.05 and c >= len(d) - 25:
            neck = (L[a:b + 1].min() + L[b:c + 1].min()) / 2
            out.append(("Head and shoulders terkonfirmasi" if last < neck else "Head and shoulders potensial", "turun", "pembalikan"))
    if len(lo) >= 3:
        a, b, c = lo[-3], lo[-2], lo[-1]
        if L[b] < L[a] and L[b] < L[c] and abs(L[a] / L[c] - 1) <= 0.05 and c >= len(d) - 25:
            neck = (H[a:b + 1].max() + H[b:c + 1].max()) / 2
            out.append(("Inverse head and shoulders terkonfirmasi" if last > neck else "Inverse head and shoulders potensial", "naik", "pembalikan"))
    # Breakout / breakdown 20 bar dengan volume
    hh, ll = H[-21:-1].max(), L[-21:-1].min()
    vr = V[-1] / V[-21:-1].mean() if V[-21:-1].mean() > 0 else np.nan
    if last > hh: out.append(("Breakout 20 bar" + (" + volume tinggi" if vr >= 1.5 else ""), "naik", "kelanjutan"))
    if last < ll: out.append(("Breakdown 20 bar" + (" + volume tinggi" if vr >= 1.5 else ""), "turun", "kelanjutan"))
    # Bull / bear flag sederhana: tiang >= 8% dalam 10 bar, lalu konsolidasi sempit 5-12 bar
    for flag_len in (6, 10):
        if len(d) > flag_len + 12:
            pole_end = len(d) - flag_len - 1
            pole = C[pole_end] / C[pole_end - 10] - 1
            seg = d.iloc[-flag_len:]
            width = (seg["High"].max() - seg["Low"].min()) / C[pole_end]
            retr = (C[pole_end] - seg["Low"].min()) / (C[pole_end] - C[pole_end - 10]) if C[pole_end] != C[pole_end - 10] else 9
            if pole >= 0.08 and width <= 0.06 and 0 <= retr <= 0.4:
                out.append(("Bull flag potensial", "naik", "kelanjutan")); break
            if pole <= -0.08 and width <= 0.06:
                rb = (seg["High"].max() - C[pole_end]) / (C[pole_end - 10] - C[pole_end]) if C[pole_end - 10] != C[pole_end] else 9
                if 0 <= rb <= 0.4:
                    out.append(("Bear flag potensial", "turun", "kelanjutan")); break
    # Struktur tren dari pivot
    if len(hi) >= 2 and len(lo) >= 2:
        if H[hi[-1]] > H[hi[-2]] and L[lo[-1]] > L[lo[-2]]: out.append(("Struktur tren naik (HH dan HL)", "naik", "tren"))
        if H[hi[-1]] < H[hi[-2]] and L[lo[-1]] < L[lo[-2]]: out.append(("Struktur tren turun (LH dan LL)", "turun", "tren"))
    return out

# ---------- Gabungan per simbol ----------
def analyze(df):
    df = df.sort_values("Date").reset_index(drop=True)
    c = df["Close"]
    r = {"tanggal": str(df["Date"].iloc[-1])[:10], "bar": len(df), "close": c.iloc[-1]}
    for n in (20, 50, 200): r[f"sma{n}"] = sma(c, n).iloc[-1]
    r["wma20"] = wma(c, 20).iloc[-1]
    for n in (12, 26, 50): r[f"ema{n}"] = ema(c, n).iloc[-1]
    macd = ema(c, 12) - ema(c, 26)
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    r["macd"], r["macd_sinyal"] = macd.iloc[-1], sig.iloc[-1]
    r["rsi14"] = rsi(c).iloc[-1]
    r["atr14"] = atr(df).iloc[-1]
    m20, s20 = sma(c, 20), c.rolling(20).std()
    r["bb_posisi"] = (c.iloc[-1] - (m20 - 2 * s20).iloc[-1]) / ((4 * s20).iloc[-1]) if s20.iloc[-1] else np.nan
    v = df["Volume"]
    r["vol_rasio"] = v.iloc[-1] / v.iloc[-21:-1].mean() if len(v) > 21 and v.iloc[-21:-1].mean() > 0 else np.nan
    r["support20"], r["resistance20"] = df["Low"].tail(20).min(), df["High"].tail(20).max()
    sig_list = []
    s50, s200 = sma(c, 50), sma(c, 200)
    if len(df) >= 205:
        diff = (s50 - s200).dropna().tail(6)
        if (diff.iloc[0] < 0) and (diff.iloc[-1] > 0): sig_list.append("Golden cross SMA50/200")
        if (diff.iloc[0] > 0) and (diff.iloc[-1] < 0): sig_list.append("Death cross SMA50/200")
    e12, e26 = ema(c, 12), ema(c, 26)
    dd = (e12 - e26).dropna().tail(4)
    if len(dd) == 4:
        if dd.iloc[0] < 0 < dd.iloc[-1]: sig_list.append("EMA12 memotong naik EMA26")
        if dd.iloc[0] > 0 > dd.iloc[-1]: sig_list.append("EMA12 memotong turun EMA26")
    if r["rsi14"] == r["rsi14"]:
        if r["rsi14"] >= 70: sig_list.append("RSI jenuh beli (>=70)")
        if r["rsi14"] <= 30: sig_list.append("RSI jenuh jual (<=30)")
    if r["sma200"] == r["sma200"]: sig_list.append("Di atas SMA200" if c.iloc[-1] > r["sma200"] else "Di bawah SMA200")
    elif r["sma50"] == r["sma50"]: sig_list.append("Di atas SMA50" if c.iloc[-1] > r["sma50"] else "Di bawah SMA50")
    r["sinyal"] = "; ".join(sig_list)
    cp, gp = candle_patterns(df), chart_patterns(df)
    r["pola_candle"] = "; ".join(f"{a} ({b}, {j})" for a, b, j in cp)
    r["pola_grafik"] = "; ".join(f"{a} ({b}, {j})" for a, b, j in gp)
    r["catatan"] = "" if len(df) >= 200 else f"Data {len(df)} bar; SMA200 butuh 200 bar"
    return r

def main():
    rows = []
    for path in sorted(glob.glob("data/*.csv")):
        name = os.path.basename(path)[:-4]
        if name.startswith("_"): continue
        try:
            df = pd.read_csv(path, parse_dates=["Date"]).dropna(subset=["Close"])
            if len(df) < 10: continue
            r = analyze(df); r["simbol"] = name; rows.append(r)
        except Exception as e:
            print(f"{name}: gagal ({e})")
    if rows:
        out = pd.DataFrame(rows)
        out = out[["simbol"] + [c for c in out.columns if c != "simbol"]].round(4)
        out.to_csv("data/_indikator.csv", index=False)
        print(f"{len(rows)} simbol dihitung")

if __name__ == "__main__":
    main()
