#!/usr/bin/env python3
"""
KRONOS WEB APP
==============
Run once: python kronos_app.py
Then open: http://localhost:5000
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, sys, threading, webbrowser, urllib.parse, requests
from datetime import datetime, timezone

CLAUDE_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
GROK_KEY    = os.environ.get("XAI_API_KEY", "")
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")
BINANCE     = "https://api.binance.com/api/v3"
PORT        = 5000
CODE_DIR    = "C:/kronos/code"
MODEL_DIR   = "C:/kronos/model"

# try loading Kronos model at startup
KRONOS_PREDICTOR = None
def load_kronos_model():
    global KRONOS_PREDICTOR
    try:
        sys.path.insert(0, CODE_DIR)
        import torch, numpy as np, pandas as pd
        from model import Kronos, KronosTokenizer, KronosPredictor
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir=MODEL_DIR)
        model     = Kronos.from_pretrained("NeoQuasar/Kronos-base", cache_dir=MODEL_DIR)
        KRONOS_PREDICTOR = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        print(f"  Kronos-base: OK (102M parameters)")
        return True
    except Exception as e:
        print(f"  Kronos-base: NOT AVAILABLE ({e})")
        return False

MS_MAP   = {"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"1d":86400000}
IV_LABEL = {"5m":"next 2h","15m":"next 6h","30m":"next 12h","1h":"next 24h","4h":"next 4d","1d":"next 2w"}
KUCOIN   = "https://api.kucoin.com/api/v1"

# Coins that use KuCoin instead of Binance
KUCOIN_SYMBOLS = {"CPOOLUSDT", "NPCUSDT", "TELUSDT"}

# ── DATA ──────────────────────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit=80):
    if symbol in KUCOIN_SYMBOLS:
        return fetch_klines_kucoin(symbol, interval, limit)
    r = requests.get(f"{BINANCE}/klines",
                     params={"symbol":symbol,"interval":interval,"limit":limit},timeout=15)
    r.raise_for_status()
    return [{"t":k[0],"o":float(k[1]),"h":float(k[2]),"l":float(k[3]),"c":float(k[4]),"v":float(k[5])} for k in r.json()]

def fetch_klines_kucoin(symbol, interval, limit=80):
    # KuCoin uses different interval format and symbol format
    iv_map = {"5m":"5min","15m":"15min","30m":"30min","1h":"1hour","4h":"4hour","1d":"1day"}
    kc_iv  = iv_map.get(interval, "1hour")
    # KuCoin symbol format: CPOOL-USDT
    kc_sym = symbol.replace("USDT", "-USDT")
    r = requests.get(f"{KUCOIN}/market/candles",
                     params={"symbol":kc_sym,"type":kc_iv},timeout=15)
    r.raise_for_status()
    data = r.json()["data"][:limit]
    # KuCoin returns [time, open, close, high, low, volume, turnover] newest first
    result = []
    for k in reversed(data):
        result.append({
            "t": int(k[0])*1000,
            "o": float(k[1]),
            "h": float(k[3]),
            "l": float(k[4]),
            "c": float(k[2]),
            "v": float(k[5])
        })
    return result

def fetch_ticker(symbol):
    if symbol in KUCOIN_SYMBOLS:
        return fetch_ticker_kucoin(symbol)
    r = requests.get(f"{BINANCE}/ticker/24hr",params={"symbol":symbol},timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_ticker_kucoin(symbol):
    kc_sym = symbol.replace("USDT", "-USDT")
    r = requests.get(f"{KUCOIN}/market/stats",params={"symbol":kc_sym},timeout=15)
    r.raise_for_status()
    d = r.json()["data"]
    last  = float(d["last"])
    open_ = float(d["open"]) if d["open"] else last
    chg   = ((last - open_) / open_ * 100) if open_ else 0
    return {
        "lastPrice":          str(last),
        "priceChangePercent": str(round(chg, 2)),
        "highPrice":          str(d.get("high") or last),
        "lowPrice":           str(d.get("low")  or last),
        "quoteVolume":        str(d.get("volValue") or 0),
    }

# ── SHARED PROMPT BUILDER ─────────────────────────────────────────────────────
def build_ai_prompt(symbol, interval, klines, ticker):
    last20 = klines[-20:]
    rows = []
    for i,k in enumerate(last20):
        chg = ((k["c"]-last20[i-1]["c"])/last20[i-1]["c"]*100) if i>0 else 0.0
        rows.append(f"[{i+1:02d}] O:{k['o']:.4f} H:{k['h']:.4f} L:{k['l']:.4f} C:{k['c']:.4f} chg:{chg:+.2f}%")
    system = """You are an expert quantitative crypto analyst. Analyze OHLCV candlestick data and return ONLY a valid JSON object. No markdown, no backticks, no explanation.
Schema: {"bullish_probability":<0-100>,"volatility_probability":<0-100>,"trend":"bullish"|"bearish"|"neutral","sentiment":"strong buy"|"buy"|"neutral"|"sell"|"strong sell","key_levels":{"support_1":<n>,"support_2":<n>,"resistance_1":<n>,"resistance_2":<n>},"forecast_candles":[{"mean":<n>,"high":<n>,"low":<n>}],"rsi_estimate":<0-100>,"macd_signal":"bullish"|"bearish"|"neutral","risk_level":"low"|"medium"|"high"|"extreme","target_price":<n>,"stop_loss":<n>,"analysis":"<3-4 sentences>","short_summary":"<1 sentence>"}
forecast_candles must have exactly 8 entries. Return ONLY the JSON."""
    user = (f"Symbol:{symbol} Interval:{interval} Price:{float(ticker['lastPrice']):.4f}\n"
            f"24h chg:{float(ticker['priceChangePercent']):.2f}% H:{float(ticker['highPrice']):.4f} L:{float(ticker['lowPrice']):.4f}\n"
            f"Last 20 candles:\n"+"\n".join(rows)+f"\nForecast {IV_LABEL[interval]}. JSON only.")
    return system, user

# ── GPT-4o ────────────────────────────────────────────────────────────────────
def run_gpt4o(symbol, interval, klines, ticker):
    system, user = build_ai_prompt(symbol, interval, klines, ticker)
    r = requests.post("https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"},
        json={"model":"gpt-4o",
              "messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.25,"max_tokens":1500,
              "response_format":{"type":"json_object"}},
        timeout=30)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"].strip())

# ── GROK-3 (xAI) ─────────────────────────────────────────────────────────────
def run_grok2(symbol, interval, klines, ticker):
    system, user = build_ai_prompt(symbol, interval, klines, ticker)
    r = requests.post("https://api.x.ai/v1/chat/completions",
        headers={"Authorization":f"Bearer {GROK_KEY}","Content-Type":"application/json"},
        json={"model":"grok-3",
              "messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.25,"max_tokens":1500},
        timeout=30)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json","").replace("```","").strip()
    return json.loads(raw)

# ── GEMINI 2.5 FLASH ─────────────────────────────────────────────────────────
def run_gemini(symbol, interval, klines, ticker):
    system, user = build_ai_prompt(symbol, interval, klines, ticker)
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        headers={"Content-Type":"application/json",
                 "x-goog-api-key": GEMINI_KEY},
        json={"system_instruction":{"parts":[{"text":system}]},
              "contents":[{"role":"user","parts":[{"text":user}]}],
              "generationConfig":{"temperature":0.25,"maxOutputTokens":8192}},
        timeout=60)
    if not r.ok:
        raise Exception(f"Gemini {r.status_code}: {r.text[:500]}")
    raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = raw.replace("```json","").replace("```","").strip()
    # find first complete JSON object
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start >= 0 and end > start:
        raw = raw[start:end]
    return json.loads(raw)

# ── CLAUDE HAIKU 4.5 ──────────────────────────────────────────────────────────
def run_claude(symbol, interval, klines, ticker):
    last20 = klines[-20:]
    rows = []
    for i,k in enumerate(last20):
        chg = ((k["c"]-last20[i-1]["c"])/last20[i-1]["c"]*100) if i>0 else 0.0
        rows.append(f"[{i+1:02d}] O:{k['o']:.4f} H:{k['h']:.4f} L:{k['l']:.4f} C:{k['c']:.4f} chg:{chg:+.2f}%")

    system = """You are an expert quantitative crypto analyst with deep knowledge of technical analysis.
Analyze the OHLCV candlestick data provided and return ONLY a valid JSON object.
No markdown, no backticks, no explanation — just the raw JSON.

Required schema:
{"bullish_probability":<integer 0-100>,"volatility_probability":<integer 0-100>,
"trend":"bullish"|"bearish"|"neutral",
"sentiment":"strong buy"|"buy"|"neutral"|"sell"|"strong sell",
"key_levels":{"support_1":<number>,"support_2":<number>,"resistance_1":<number>,"resistance_2":<number>},
"forecast_candles":[{"mean":<number>,"high":<number>,"low":<number>}],
"rsi_estimate":<integer 0-100>,
"macd_signal":"bullish"|"bearish"|"neutral",
"risk_level":"low"|"medium"|"high"|"extreme",
"target_price":<number>,
"stop_loss":<number>,
"analysis":"<3-4 detailed sentences covering trend structure, momentum, key risks and opportunity>",
"short_summary":"<1 concise actionable sentence>"}
forecast_candles must have exactly 8 entries. Return ONLY the JSON object."""

    user = (f"Symbol: {symbol} | Interval: {interval}\n"
            f"Current price: {float(ticker['lastPrice']):.4f}\n"
            f"24h change: {float(ticker['priceChangePercent']):.2f}% | "
            f"High: {float(ticker['highPrice']):.4f} | Low: {float(ticker['lowPrice']):.4f}\n"
            f"24h volume: {float(ticker['quoteVolume']):.0f} USDT\n\n"
            f"Last 20 candles:\n"+"\n".join(rows)+
            f"\n\nForecast {IV_LABEL[interval]}. Return only the JSON.")

    r = requests.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key":CLAUDE_KEY,
                 "anthropic-version":"2023-06-01",
                 "content-type":"application/json"},
        json={"model":"claude-haiku-4-5",
              "max_tokens":1500,
              "system":system,
              "messages":[{"role":"user","content":user}]},
        timeout=30)
    r.raise_for_status()
    raw = r.json()["content"][0]["text"].strip()
    raw = raw.replace("```json","").replace("```","").strip()
    return json.loads(raw)

# ── KRONOS-BASE REAL MODEL ────────────────────────────────────────────────────
def run_kronos_model(symbol, interval, klines, ticker):
    import numpy as np, pandas as pd
    if not KRONOS_PREDICTOR:
        raise Exception("Kronos model not loaded")

    df = pd.DataFrame([{"open":k["o"],"high":k["h"],"low":k["l"],"close":k["c"],"volume":k["v"]} for k in klines])
    ms     = MS_MAP[interval]
    last_t = klines[-1]["t"]
    x_times = pd.Series([datetime.fromtimestamp(k["t"]/1000, tz=timezone.utc) for k in klines])
    y_times = pd.Series([datetime.fromtimestamp((last_t+ms*(i+1))/1000, tz=timezone.utc) for i in range(8)])

    result = KRONOS_PREDICTOR.predict(df=df, x_timestamp=x_times, y_timestamp=y_times, pred_len=8)
    fc = []
    for i in range(len(result)):
        row  = result.iloc[i]
        mean = float(row.get("close", row.iloc[0]))
        high = float(row.get("high",  mean*1.005))
        low  = float(row.get("low",   mean*0.995))
        fc.append({"mean":round(mean,4),"high":round(high,4),"low":round(low,4)})

    # technical indicators
    closes = [k["c"] for k in klines]
    gains,losses = [],[]
    for i in range(1,min(15,len(closes))):
        d=closes[i]-closes[i-1]; gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains)/len(gains) if gains else 0
    al=sum(losses)/len(losses) if losses else 0.001
    rsi = round(100-(100/(1+ag/al)),1)
    trend = "bullish" if closes[-1]>closes[-5] else "bearish" if closes[-1]<closes[-5] else "neutral"
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,min(11,len(closes)))]
    vol_pct = round(float(np.std(returns))*100,2) if returns else 0
    recent  = closes[-20:]
    bull = max(10,min(90, 50+(15 if trend=="bullish" else -15 if trend=="bearish" else 0)+(20 if rsi<30 else -20 if rsi>70 else 0)))
    tgt  = fc[-1]["mean"] if fc else closes[-1]
    sl   = round(min(recent)*0.998,4)

    return {
        "bullish_probability":    bull,
        "volatility_probability": min(90,int(vol_pct*20)),
        "trend":      trend,
        "sentiment":  "buy" if bull>60 else "sell" if bull<40 else "neutral",
        "key_levels": {"support_1":round(min(recent)*0.998,4),"support_2":round(min(recent)*0.995,4),
                       "resistance_1":round(max(recent)*1.002,4),"resistance_2":round(max(recent)*1.005,4)},
        "forecast_candles": fc,
        "rsi_estimate":  rsi,
        "macd_signal":   trend,
        "risk_level":    "high" if vol_pct>2 else "medium" if vol_pct>0.8 else "low",
        "target_price":  tgt,
        "stop_loss":     sl,
        "analysis": f"Kronos-base model (102M parameters, trained on 12B+ candlesticks from 45 global exchanges) projects price toward {tgt:.4f} over {IV_LABEL[interval]}. RSI is {rsi} indicating {'overbought' if rsi>70 else 'oversold' if rsi<30 else 'neutral'} conditions. Volatility is {vol_pct}% classified as {'high' if vol_pct>2 else 'medium' if vol_pct>0.8 else 'low'} risk. Key support at {sl:.4f}.",
        "short_summary": f"Kronos-base forecasts {trend} move toward {tgt:.4f} — {IV_LABEL[interval]}."
    }

# ── HTML PAGE ─────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kronos — Crypto AI Forecast</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#e2e8f0;min-height:100vh}
.topbar{background:#161b27;border-bottom:1px solid #21293d;padding:14px 24px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.logo{font-size:18px;font-weight:700;color:#f1f5f9;letter-spacing:-0.3px}
.logo span{color:#378ADD}
select{background:#0d1117;color:#e2e8f0;border:1px solid #2d3748;border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer;outline:none}
select:hover{border-color:#4a5568}
.run-btn{background:#378ADD;color:#fff;border:none;border-radius:8px;padding:8px 22px;font-size:13px;font-weight:600;cursor:pointer;transition:background .15s;white-space:nowrap}
.run-btn:hover{background:#2563eb}
.run-btn:disabled{background:#2d3748;color:#64748b;cursor:not-allowed}
.main{padding:20px 24px;max-width:1100px;margin:0 auto}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.met{background:#161b27;border-radius:10px;padding:13px 15px;border:1px solid #21293d}
.ml{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px}
.mv{font-size:19px;font-weight:700;color:#f1f5f9}
.ms{font-size:12px;margin-top:2px;color:#64748b}
.up{color:#1D9E75}.dn{color:#E24B4A}
.cbox{background:#161b27;border-radius:10px;padding:16px;border:1px solid #21293d;margin-bottom:14px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}
.pbar{background:#1e2d40;border-radius:6px;height:8px;overflow:hidden;margin:6px 0 3px}
.pf{height:100%;border-radius:6px;transition:width 1s ease}
.pval{font-size:21px;font-weight:700}
.lvg{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:9px}
.lv{background:#0d1117;border-radius:8px;padding:8px 11px}
.lvl{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.5px}
.lvv{font-size:14px;font-weight:700;margin-top:2px}
.abox{background:#0d1117;border-radius:10px;padding:14px;font-size:14px;line-height:1.75;color:#cbd5e1;border-left:3px solid #378ADD;margin-top:10px}
.summary{background:#0f2d1a;border:1px solid #1D9E75;border-radius:10px;padding:12px 15px;font-size:14px;font-weight:500;color:#a7f3d0;margin-bottom:14px}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;margin-right:5px}
.legend{display:flex;gap:12px;flex-wrap:wrap}
.li{display:flex;align-items:center;gap:5px;font-size:11px;color:#64748b}
.ld{width:16px;height:3px;border-radius:2px}
.spin{display:none;width:16px;height:16px;border:2px solid #2d3748;border-top-color:#378ADD;border-radius:50%;animation:sp .7s linear infinite;margin-left:8px}
@keyframes sp{to{transform:rotate(360deg)}}
.status{font-size:12px;color:#64748b;margin-left:8px}
.empty{text-align:center;padding:60px 20px;color:#475569;font-size:14px}
.tc{background:#161b27;border-radius:10px;padding:12px 15px;border:1px solid #21293d}
@media(max-width:600px){.metrics{grid-template-columns:1fr 1fr}.g2{grid-template-columns:1fr}.g3{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">Kronos <span>AI</span></div>
  <select id="sym">
    <option value="BTCUSDT">BTC / USDT — Bitcoin</option>
    <option value="ETHUSDT">ETH / USDT — Ethereum</option>
    <option value="SOLUSDT">SOL / USDT — Solana</option>
    <option value="BNBUSDT">BNB / USDT — BNB</option>
    <option value="XRPUSDT">XRP / USDT — XRP</option>
    <option value="ADAUSDT">ADA / USDT — Cardano</option>
    <option value="TAOUSDT">TAO / USDT — Bittensor</option>
    <option value="ZECUSDT">ZEC / USDT — Zcash</option>
    <option value="DOGEUSDT">DOGE / USDT — Dogecoin</option>
    <option value="AVAXUSDT">AVAX / USDT — Avalanche</option>
    <option value="DOTUSDT">DOT / USDT — Polkadot</option>
    <option value="LINKUSDT">LINK / USDT — Chainlink</option>
    <option value="UNIUSDT">UNI / USDT — Uniswap</option>
    <option value="CPOOLUSDT">CPOOL / USDT — Clearpool ★KuCoin</option>
    <option value="NPCUSDT">NPC / USDT — Non Playable Coin ★KuCoin</option>
    <option value="TELUSDT">TEL / USDT — Telcoin ★KuCoin</option>
  </select>
  <select id="iv">
    <option value="5m">5 minutes — scalping</option>
    <option value="15m" selected>15 minutes — day trading</option>
    <option value="30m">30 minutes — day trading</option>
    <option value="1h">1 hour</option>
    <option value="4h">4 hours</option>
    <option value="1d">1 day</option>
  </select>
  <button class="run-btn" id="runBtn" onclick="runForecast()">Run AI Forecast</button>
  <select id="engine" title="AI Engine">
    <option value="gpt4o">GPT-4o (OpenAI)</option>
    <option value="grok2">Grok-3 (xAI)</option>
    <option value="gemini">Gemini 2.5 Flash (Google)</option>
    <option value="claude">Claude Haiku 4.5 (Anthropic)</option>
    <option value="kronos">Kronos-base 102M (real model)</option>
    <option value="combined">⚡ Combined — All 5 engines</option>
  </select>
  <div class="spin" id="spin"></div>
  <span class="status" id="status"></span>
</div>

<div class="main">
  <div id="content">
    <div class="empty">Select a symbol and interval above, then click <b>Run AI Forecast</b></div>
  </div>
</div>

<script>
let chart = null;

async function runForecast() {
  const sym    = document.getElementById('sym').value;
  const iv     = document.getElementById('iv').value;
  const engine = document.getElementById('engine').value;
  const btn    = document.getElementById('runBtn');
  const spin   = document.getElementById('spin');
  const status = document.getElementById('status');

  btn.disabled = true;
  spin.style.display = 'block';

  if (engine === 'combined') {
    status.textContent = 'Running all 5 engines — please wait (~30 seconds)...';
    try {
      const res = await fetch(`/forecast_combined?symbol=${sym}&interval=${iv}`);
      const data = await res.json();
      if (data.error) {
        status.textContent = 'Error: ' + data.error;
        btn.disabled = false; spin.style.display = 'none'; return;
      }
      renderCombined(data);
      status.textContent = `Combined consensus — ${new Date().toLocaleTimeString()}`;
    } catch(e) {
      status.textContent = 'Error: ' + e.message;
    }
    btn.disabled = false; spin.style.display = 'none'; return;
  }

  status.textContent = 'Fetching market data...';

  try {
    status.textContent = 'Running AI analysis...';
    const res = await fetch(`/forecast?symbol=${sym}&interval=${iv}&engine=${engine}`);
    const data = await res.json();

    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      btn.disabled = false;
      spin.style.display = 'none';
      return;
    }

    renderDashboard(data);
    status.textContent = `Updated ${new Date().toLocaleTimeString()} · ${data.engine}`;
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }

  btn.disabled = false;
  spin.style.display = 'none';
}

function fp(v) {
  if (v >= 10000) return '$' + v.toLocaleString('en-US', {maximumFractionDigits:0});
  if (v >= 1)     return '$' + v.toFixed(2);
  return '$' + v.toFixed(5);
}
function fvol(v) {
  if (v >= 1e9) return (v/1e9).toFixed(1)+'B';
  if (v >= 1e6) return (v/1e6).toFixed(0)+'M';
  return (v/1e3).toFixed(0)+'K';
}

function renderDashboard(d) {
  const a = d.analysis;
  const price = d.price, chg = d.chg;
  const bull = a.bullish_probability || 50;
  const volp = a.volatility_probability || 50;
  const trend = (a.trend || 'neutral').toUpperCase();
  const sent  = (a.sentiment || 'neutral').toUpperCase();
  const risk  = (a.risk_level || 'medium').toUpperCase();
  const rsi   = a.rsi_estimate || 50;
  const tgt   = a.target_price || price;
  const sl    = a.stop_loss    || price;
  const kl    = a.key_levels   || {};
  const s1 = kl.support_1    || price*0.97;
  const s2 = kl.support_2    || price*0.94;
  const r1 = kl.resistance_1 || price*1.03;
  const r2 = kl.resistance_2 || price*1.06;
  const rr = Math.abs(sl-price)>0 ? '1 : '+(Math.abs((tgt-price)/(sl-price))).toFixed(1) : '—';
  const tgt_p = ((tgt-price)/price*100).toFixed(2);
  const sl_p  = ((sl-price)/price*100).toFixed(2);

  const cc = chg>=0?'#1D9E75':'#E24B4A';
  const tc = trend=='BULLISH'?'#1D9E75':trend=='BEARISH'?'#E24B4A':'#888';
  const sc = sent.includes('BUY')?'#1D9E75':sent.includes('SELL')?'#E24B4A':'#888';
  const rc = {LOW:'#1D9E75',MEDIUM:'#BA7517',HIGH:'#E24B4A',EXTREME:'#A32D2D'}[risk]||'#888';
  const bc = bull>=50?'#1D9E75':'#E24B4A';
  const rsic = rsi>70?'#E24B4A':rsi<30?'#1D9E75':'#f1f5f9';
  const tbg = trend=='BULLISH'?'#0a2e18':trend=='BEARISH'?'#2e0a0a':'#1a1a1a';
  const sbg = sent.includes('BUY')?'#0a2e18':sent.includes('SELL')?'#2e0a0a':'#1a1a1a';

  // build chart data
  const closes = d.klines.map(k=>k.c);
  const times  = d.klines.map(k=>k.tl);
  const fc     = a.forecast_candles || [];
  const fcm    = fc.map(c=>c.mean);
  const fch    = fc.map(c=>c.high);
  const fcl2   = fc.map(c=>c.low);
  const fct    = d.fc_times;
  const allT   = [...times,...fct];
  const hist   = [...closes,...Array(fc.length).fill(null)];
  const pad    = Array(closes.length-1).fill(null);
  const fcmD   = [...pad,closes[closes.length-1],...fcm];
  const fchD   = [...pad,closes[closes.length-1],...fch];
  const fclD   = [...pad,closes[closes.length-1],...fcl2];

  document.getElementById('content').innerHTML = `
<div class="summary">${a.short_summary||''}</div>

<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap">
  <span style="font-size:15px;font-weight:700;color:#f1f5f9">${d.symbol} &nbsp;·&nbsp; ${d.interval.toUpperCase()}</span>
  <span class="badge" style="background:${tbg};color:${tc}">${trend}</span>
  <span class="badge" style="background:${sbg};color:${sc}">${sent}</span>
  <span class="badge" style="background:#1a1a1a;color:${rc}">RISK: ${risk}</span>
</div>

<div class="metrics">
  <div class="met"><div class="ml">Price</div><div class="mv">${fp(price)}</div><div class="ms" style="color:${cc}">${chg>=0?'+':''}${chg.toFixed(2)}% 24h</div></div>
  <div class="met"><div class="ml">Volume 24h</div><div class="mv">${fvol(d.vol)}</div><div class="ms">USDT</div></div>
  <div class="met"><div class="ml">24h H / L</div><div class="mv" style="font-size:16px">${fp(d.high24)}</div><div class="ms">${fp(d.low24)}</div></div>
  <div class="met"><div class="ml">RSI / MACD</div><div class="mv" style="color:${rsic}">${rsi}</div><div class="ms">${a.macd_signal||'—'}</div></div>
</div>

<div class="cbox">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px">
    <span style="font-size:13px;color:#94a3b8;font-weight:600">${d.symbol} ${d.interval.toUpperCase()} — ${d.iv_label}</span>
    <div class="legend">
      <span class="li"><span class="ld" style="background:#378ADD"></span>Historical</span>
      <span class="li"><span class="ld" style="background:#1D9E75"></span>AI forecast</span>
      <span class="li"><span class="ld" style="background:rgba(29,158,117,0.4)"></span>Range</span>
    </div>
  </div>
  <div style="position:relative;width:100%;height:300px">
    <canvas id="mc" role="img" aria-label="price chart"></canvas>
  </div>
</div>

<div class="g2">
  <div class="cbox" style="margin-bottom:0">
    <div class="ml">Bullish probability</div>
    <div class="pbar"><div class="pf" id="bb" style="width:0%;background:${bc}"></div></div>
    <div class="pval" style="color:${bc}">${bull}%</div>
    <div style="margin-top:12px">
      <div class="ml">Elevated volatility</div>
      <div class="pbar"><div class="pf" id="vb" style="width:0%;background:#BA7517"></div></div>
      <div class="pval" style="color:#BA7517">${volp}%</div>
    </div>
  </div>
  <div class="cbox" style="margin-bottom:0">
    <div class="ml">Key price levels</div>
    <div class="lvg">
      <div class="lv" style="border-left:3px solid #E24B4A"><div class="lvl">Resistance 2</div><div class="lvv" style="color:#E24B4A">${fp(r2)}</div></div>
      <div class="lv" style="border-left:3px solid #f97316"><div class="lvl">Resistance 1</div><div class="lvv" style="color:#f97316">${fp(r1)}</div></div>
      <div class="lv" style="border-left:3px solid #1D9E75"><div class="lvl">Support 1</div><div class="lvv" style="color:#1D9E75">${fp(s1)}</div></div>
      <div class="lv" style="border-left:3px solid #0F6E56"><div class="lvl">Support 2</div><div class="lvv" style="color:#0F6E56">${fp(s2)}</div></div>
    </div>
  </div>
</div>

<div class="g3" style="margin-top:12px">
  <div class="tc"><div class="ml">Price target</div><div style="font-size:19px;font-weight:700;color:#1D9E75;margin-top:4px">${fp(tgt)}</div><div class="ms">${tgt_p>=0?'+':''}${tgt_p}% from now</div></div>
  <div class="tc"><div class="ml">Stop loss</div><div style="font-size:19px;font-weight:700;color:#E24B4A;margin-top:4px">${fp(sl)}</div><div class="ms">${sl_p}% from now</div></div>
  <div class="tc"><div class="ml">Risk / Reward</div><div style="font-size:19px;font-weight:700;color:#f1f5f9;margin-top:4px">${rr}</div><div class="ms">target vs stop</div></div>
</div>

<div class="cbox" style="margin-top:12px">
  <span style="font-size:13px;color:#94a3b8;font-weight:600">AI analysis</span>
  <div class="abox">${a.analysis||''}</div>
</div>`;

  // animate bars
  setTimeout(()=>{
    document.getElementById('bb').style.width = bull+'%';
    document.getElementById('vb').style.width = volp+'%';
  },100);

  // draw chart
  if (chart) { chart.destroy(); chart = null; }
  const ctx = document.getElementById('mc').getContext('2d');
  chart = new Chart(ctx, {
    type:'line',
    data:{labels:allT,datasets:[
      {label:'Historical',data:hist,borderColor:'#378ADD',backgroundColor:'rgba(55,138,221,0.07)',borderWidth:2,pointRadius:0,pointHoverRadius:3,fill:true,tension:0.3},
      {label:'High',data:fchD,borderColor:'rgba(29,158,117,0.25)',backgroundColor:'rgba(29,158,117,0.12)',borderWidth:1,pointRadius:0,fill:'+1',tension:0.3},
      {label:'Low',data:fclD,borderColor:'rgba(29,158,117,0.25)',borderWidth:1,pointRadius:0,fill:false,tension:0.3},
      {label:'Forecast',data:fcmD,borderColor:'#1D9E75',borderWidth:2,borderDash:[6,3],pointRadius:0,pointHoverRadius:4,fill:false,tension:0.3}
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e2330',borderColor:'#2d3748',borderWidth:1,titleColor:'#94a3b8',bodyColor:'#e2e8f0',
        callbacks:{label:c=>c.parsed.y===null?null:` ${c.dataset.label}: ${fp(c.parsed.y)}`}}},
      scales:{
        x:{ticks:{maxTicksLimit:12,color:'#475569',font:{size:11},autoSkip:true,maxRotation:45},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{ticks:{color:'#475569',font:{size:11},callback:v=>v>=1000?'$'+v.toLocaleString('en-US',{maximumFractionDigits:0}):'$'+v.toFixed(4)},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}
function renderCombined(d) {
  const engines = d.engines;
  const cons    = d.consensus;
  const price   = d.price;
  const chg     = d.chg;
  const cc      = chg>=0?'#1D9E75':'#E24B4A';

  // consensus colors
  const votes   = cons.bullish_votes;
  const total   = cons.total;
  const cvote   = votes >= Math.ceil(total/2) ? '#1D9E75' : votes === 0 ? '#E24B4A' : '#BA7517';
  const clabel  = `${votes}/${total} ${votes >= Math.ceil(total/2) ? 'BULLISH' : votes === 0 ? 'BEARISH' : 'BULLISH'}`;
  const csignal = votes >= Math.ceil(total/2) ? 'BUY' : votes === 0 ? 'SELL' : 'NEUTRAL';
  const csbg    = votes >= 2 ? '#0a2e18' : votes === 0 ? '#2e0a0a' : '#1a1a1a';

  // build chart data from Kronos (most specialized)
  const kronosData = engines.find(e=>e.name==='Kronos') || engines[0];
  const closes  = d.klines.map(k=>k.c);
  const times   = d.klines.map(k=>k.tl);
  const fc      = kronosData.forecast_candles || [];
  const fcm     = fc.map(c=>c.mean);
  const fch     = fc.map(c=>c.high);
  const fcl     = fc.map(c=>c.low);
  const fct     = d.fc_times;
  const allT    = [...times,...fct];
  const hist    = [...closes,...Array(fc.length).fill(null)];
  const pad     = Array(closes.length-1).fill(null);
  const fcmD    = [...pad,closes[closes.length-1],...fcm];
  const fchD    = [...pad,closes[closes.length-1],...fch];
  const fclD    = [...pad,closes[closes.length-1],...fcl];

  function fp(v){
    if(v>=10000) return '$'+v.toLocaleString('en-US',{maximumFractionDigits:0});
    if(v>=1) return '$'+v.toFixed(2);
    return '$'+v.toFixed(5);
  }

  // engine cards HTML
  const engineCards = engines.map(e => {
    const ec = e.bullish_probability>=50?'#1D9E75':'#E24B4A';
    const et = (e.trend||'neutral').toUpperCase();
    const etbg = et==='BULLISH'?'#0a2e18':et==='BEARISH'?'#2e0a0a':'#1a1a1a';
    const etc = et==='BULLISH'?'#1D9E75':et==='BEARISH'?'#E24B4A':'#888';
    return `<div style="background:#0d1117;border-radius:10px;padding:14px 16px;border:1px solid #21293d">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
        <span style="font-size:13px;font-weight:600;color:#94a3b8">${e.name}</span>
        <span style="font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;background:${etbg};color:${etc}">${et}</span>
      </div>
      <div style="font-size:11px;color:#475569;margin-bottom:4px">Bullish probability</div>
      <div style="background:#1e2d40;border-radius:4px;height:7px;overflow:hidden;margin-bottom:4px">
        <div style="width:${e.bullish_probability}%;height:100%;background:${ec};border-radius:4px"></div>
      </div>
      <div style="font-size:18px;font-weight:700;color:${ec}">${e.bullish_probability}%</div>
      <div style="font-size:11px;color:#475569;margin-top:8px">Target: <span style="color:#f1f5f9;font-weight:600">${fp(e.target_price)}</span></div>
      <div style="font-size:11px;color:#475569">Stop: <span style="color:#E24B4A;font-weight:600">${fp(e.stop_loss)}</span></div>
      <div style="font-size:12px;color:#64748b;margin-top:8px;line-height:1.5">${(e.short_summary||'').substring(0,120)}${(e.short_summary||'').length>120?'...':''}</div>
    </div>`;
  }).join('');

  document.getElementById('content').innerHTML = `
<div style="background:linear-gradient(135deg,#0a1628,#0f2d1a);border:1px solid ${cvote};border-radius:12px;padding:18px 20px;margin-bottom:16px">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
    <div>
      <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px">Consensus signal — ${d.symbol} ${d.interval.toUpperCase()}</div>
      <div style="font-size:28px;font-weight:700;color:${cvote}">${clabel}</div>
      <div style="font-size:13px;color:#94a3b8;margin-top:4px">${cons.agreement} — avg bullish: ${cons.avg_bullish}% — avg target: ${fp(cons.avg_target)}</div>
    </div>
    <div style="text-align:center;background:rgba(0,0,0,0.3);border-radius:10px;padding:12px 20px">
      <div style="font-size:11px;color:#475569;margin-bottom:4px">Final signal</div>
      <div style="font-size:22px;font-weight:700;color:${cvote}">${csignal}</div>
      <div style="font-size:11px;color:#475569;margin-top:4px">Risk: ${cons.risk.toUpperCase()}</div>
    </div>
  </div>
</div>

<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap">
  <span style="font-size:15px;font-weight:700;color:#f1f5f9">${d.symbol} · ${d.interval.toUpperCase()}</span>
  <span style="font-size:20px;font-weight:700;color:#f1f5f9">${fp(price)}</span>
  <span style="font-size:13px;color:${cc}">${chg>=0?'+':''}${chg.toFixed(2)}% 24h</span>
</div>

<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px">${engineCards}</div>

<div style="background:#161b27;border-radius:10px;padding:18px;border:1px solid #21293d;margin-bottom:14px">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px">
    <span style="font-size:13px;color:#94a3b8;font-weight:600">${d.symbol} ${d.interval.toUpperCase()} — Kronos-base price forecast</span>
    <div style="display:flex;gap:12px;flex-wrap:wrap">
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:#64748b"><span style="width:16px;height:3px;background:#378ADD;border-radius:2px;display:inline-block"></span>Historical</span>
      <span style="display:flex;align-items:center;gap:5px;font-size:11px;color:#64748b"><span style="width:16px;height:3px;background:#7F77DD;border-radius:2px;display:inline-block"></span>Kronos forecast</span>
    </div>
  </div>
  <div style="position:relative;width:100%;height:280px"><canvas id="mc" role="img" aria-label="chart"></canvas></div>
</div>

<div style="background:#161b27;border-radius:10px;padding:14px 16px;border:1px solid #21293d">
  <div style="font-size:13px;color:#94a3b8;font-weight:600;margin-bottom:10px">Combined analysis</div>
  <div style="background:#0d1117;border-radius:8px;padding:14px;font-size:14px;line-height:1.75;color:#cbd5e1;border-left:3px solid ${cvote}">${cons.combined_analysis}</div>
</div>`;

  if (chart) { chart.destroy(); chart = null; }
  const ctx = document.getElementById('mc').getContext('2d');
  chart = new Chart(ctx,{
    type:'line',data:{labels:allT,datasets:[
      {label:'Historical',data:hist,borderColor:'#378ADD',backgroundColor:'rgba(55,138,221,0.07)',borderWidth:2,pointRadius:0,pointHoverRadius:3,fill:true,tension:0.3},
      {label:'High',data:fchD,borderColor:'rgba(127,119,221,0.2)',backgroundColor:'rgba(127,119,221,0.08)',borderWidth:1,pointRadius:0,fill:'+1',tension:0.3},
      {label:'Low',data:fclD,borderColor:'rgba(127,119,221,0.2)',borderWidth:1,pointRadius:0,fill:false,tension:0.3},
      {label:'Kronos',data:fcmD,borderColor:'#7F77DD',borderWidth:2,borderDash:[6,3],pointRadius:0,pointHoverRadius:4,fill:false,tension:0.3}
    ]},
    options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e2330',borderColor:'#2d3748',borderWidth:1,titleColor:'#94a3b8',bodyColor:'#e2e8f0',
        callbacks:{label:c=>c.parsed.y===null?null:` ${c.dataset.label}: ${fp(c.parsed.y)}`}}},
      scales:{
        x:{ticks:{maxTicksLimit:12,color:'#475569',font:{size:11},autoSkip:true,maxRotation:45},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{ticks:{color:'#475569',font:{size:11},callback:v=>v>=1000?'$'+v.toLocaleString('en-US',{maximumFractionDigits:0}):'$'+v.toFixed(4)},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}
</script>
</body>
</html>"""

# ── SERVER ────────────────────────────────────────────────────────────────────
def time_label(ts_ms, interval):
    dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
    if interval == "1d":             return dt.strftime("%b %d")
    if interval in ("5m","15m","30m"): return dt.strftime("%H:%M")
    return dt.strftime("%m/%d %H:%M")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # suppress logs

    def send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",len(body))
        self.send_header("Access-Control-Allow-Origin","*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type","text/html")
            self.send_header("Content-Length",len(body))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/forecast_combined":
            params   = urllib.parse.parse_qs(parsed.query)
            symbol   = params.get("symbol",  ["BTCUSDT"])[0].upper()
            interval = params.get("interval",["15m"])[0]

            try:
                klines = fetch_klines(symbol, interval)
                ticker = fetch_ticker(symbol)
            except Exception as e:
                self.send_json({"error":f"Binance fetch failed: {e}"})
                return

            results = []
            errors  = []

            # Run all 3 engines
            engines_to_run = []
            if OPENAI_KEY:       engines_to_run.append(("GPT-4o",          lambda: run_gpt4o(symbol, interval, klines, ticker)))
            if GROK_KEY:         engines_to_run.append(("Grok-2",           lambda: run_grok2(symbol, interval, klines, ticker)))
            if GEMINI_KEY:       engines_to_run.append(("Gemini 1.5 Pro",   lambda: run_gemini(symbol, interval, klines, ticker)))
            if CLAUDE_KEY:       engines_to_run.append(("Claude Haiku 4.5", lambda: run_claude(symbol, interval, klines, ticker)))
            if KRONOS_PREDICTOR: engines_to_run.append(("Kronos",           lambda: run_kronos_model(symbol, interval, klines, ticker)))

            if not engines_to_run:
                self.send_json({"error":"No engines available. Set GROQ_API_KEY or ANTHROPIC_API_KEY."})
                return

            for name, fn in engines_to_run:
                try:
                    r = fn()
                    r["name"] = name
                    results.append(r)
                except Exception as e:
                    errors.append(f"{name}: {e}")

            if not results:
                self.send_json({"error":"All engines failed: " + " | ".join(errors)})
                return

            # Build consensus
            bulls   = [r["bullish_probability"] for r in results]
            trends  = [r.get("trend","neutral") for r in results]
            targets = [r.get("target_price", float(ticker["lastPrice"])) for r in results]
            stops   = [r.get("stop_loss",    float(ticker["lastPrice"])) for r in results]
            risks   = [r.get("risk_level","medium") for r in results]

            avg_bull   = round(sum(bulls)/len(bulls))
            avg_target = round(sum(targets)/len(targets), 4)
            avg_stop   = round(sum(stops)/len(stops), 4)
            bull_votes = sum(1 for b in bulls if b >= 50)
            bear_votes = sum(1 for t in trends if t == "bearish")
            bull_trend = sum(1 for t in trends if t == "bullish")

            if bull_votes == len(results):   agreement = "Strong consensus — all engines agree"
            elif bull_votes >= len(results)//2+1: agreement = "Majority bullish"
            elif bear_votes == len(results): agreement = "Strong consensus — all engines bearish"
            elif bear_votes >= len(results)//2+1: agreement = "Majority bearish"
            else:                            agreement = "Mixed signals — engines disagree"

            risk_order = {"low":0,"medium":1,"high":2,"extreme":3}
            cons_risk  = max(risks, key=lambda x: risk_order.get(x,1))

            combined_analysis = (
                f"{agreement}. "
                f"Across {len(results)} engine{'s' if len(results)>1 else ''} "
                f"({', '.join([r['name'] for r in results])}), "
                f"the average bullish probability is {avg_bull}% "
                f"with {bull_trend} bullish and {bear_votes} bearish trend signals. "
                f"Average price target: {avg_target:.4f} with stop at {avg_stop:.4f}. "
                f"Overall risk level: {cons_risk.upper()}."
            )
            if errors:
                combined_analysis += f" Note: {'; '.join(errors)}."

            ms     = MS_MAP[interval]
            last_t = klines[-1]["t"]
            # Use Kronos candles if available, else first result
            kronos_r = next((r for r in results if r["name"]=="Kronos"), results[0])
            fc_len   = len(kronos_r.get("forecast_candles",[]))

            result = {
                "symbol":   symbol,
                "interval": interval,
                "price":    float(ticker["lastPrice"]),
                "chg":      float(ticker["priceChangePercent"]),
                "klines":   [{"c":k["c"],"tl":time_label(k["t"],interval)} for k in klines],
                "fc_times": [time_label(last_t+ms*(i+1),interval) for i in range(fc_len)],
                "engines":  results,
                "consensus": {
                    "bullish_votes": bull_votes,
                    "total":         len(results),
                    "avg_bullish":   avg_bull,
                    "avg_target":    avg_target,
                    "avg_stop":      avg_stop,
                    "agreement":     agreement,
                    "risk":          cons_risk,
                    "combined_analysis": combined_analysis,
                }
            }
            self.send_json(result)
            return

        if parsed.path == "/forecast":
            params   = urllib.parse.parse_qs(parsed.query)
            symbol   = params.get("symbol",  ["BTCUSDT"])[0].upper()
            interval = params.get("interval",["15m"])[0]
            engine   = params.get("engine",  ["gpt4o"])[0]

            # key check
            if engine == "gpt4o" and not OPENAI_KEY:
                self.send_json({"error":"OPENAI_API_KEY not set. Run: setx OPENAI_API_KEY sk-... then restart."})
                return
            if engine == "grok2" and not GROK_KEY:
                self.send_json({"error":"XAI_API_KEY not set. Run: setx XAI_API_KEY xai-... then restart."})
                return
            if engine == "gemini" and not GEMINI_KEY:
                self.send_json({"error":"GEMINI_API_KEY not set. Run: setx GEMINI_API_KEY AI... then restart."})
                return
            if engine == "claude" and not CLAUDE_KEY:
                self.send_json({"error":"ANTHROPIC_API_KEY not set. Run: setx ANTHROPIC_API_KEY sk-ant-... then restart."})
                return
            if engine == "kronos" and not KRONOS_PREDICTOR:
                self.send_json({"error":"Kronos model not loaded. Make sure C:/kronos/model and C:/kronos/code exist."})
                return

            try:
                klines = fetch_klines(symbol, interval)
                ticker = fetch_ticker(symbol)
            except Exception as e:
                self.send_json({"error":f"Binance fetch failed: {e}"})
                return

            try:
                if engine == "gpt4o":
                    analysis = run_gpt4o(symbol, interval, klines, ticker)
                elif engine == "grok2":
                    analysis = run_grok2(symbol, interval, klines, ticker)
                elif engine == "gemini":
                    analysis = run_gemini(symbol, interval, klines, ticker)
                elif engine == "claude":
                    analysis = run_claude(symbol, interval, klines, ticker)
                elif engine == "kronos":
                    analysis = run_kronos_model(symbol, interval, klines, ticker)
                else:
                    self.send_json({"error":f"Unknown engine: {engine}"})
                    return
            except Exception as e:
                self.send_json({"error":f"AI analysis failed ({engine}): {e}"})
                return

            ms     = MS_MAP[interval]
            last_t = klines[-1]["t"]
            fc_len = len(analysis.get("forecast_candles",[]))

            result = {
                "symbol":   symbol,
                "interval": interval,
                "iv_label": IV_LABEL[interval],
                "engine": {"gpt4o":"GPT-4o","grok2":"Grok-3","gemini":"Gemini 2.5 Flash","claude":"Claude Haiku 4.5","kronos":"Kronos-base 102M"}.get(engine, engine),
                "price":    float(ticker["lastPrice"]),
                "chg":      float(ticker["priceChangePercent"]),
                "vol":      float(ticker["quoteVolume"]),
                "high24":   float(ticker["highPrice"]),
                "low24":    float(ticker["lowPrice"]),
                "klines":   [{"c":k["c"],"tl":time_label(k["t"],interval)} for k in klines],
                "fc_times": [time_label(last_t+ms*(i+1),interval) for i in range(fc_len)],
                "analysis": analysis,
            }
            self.send_json(result)
            return

        self.send_response(404)
        self.end_headers()

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n  {'━'*44}")
    print(f"  KRONOS WEB APP — ALL ENGINES")
    print(f"  {'━'*44}")
    if OPENAI_KEY:   print(f"  GPT-4o         : OK")
    else:            print(f"  GPT-4o         : NOT SET (setx OPENAI_API_KEY sk-...)")
    if GROK_KEY:     print(f"  Grok-2         : OK")
    else:            print(f"  Grok-2         : NOT SET (setx XAI_API_KEY xai-...)")
    if GEMINI_KEY:   print(f"  Gemini 1.5 Pro : OK")
    else:            print(f"  Gemini 1.5 Pro : NOT SET (setx GEMINI_API_KEY AI...)")
    if CLAUDE_KEY:   print(f"  Claude Haiku   : OK")
    else:            print(f"  Claude Haiku   : NOT SET (setx ANTHROPIC_API_KEY sk-ant-...)")
    print(f"  Loading Kronos-base model (may take 30 seconds)...")
    load_kronos_model()
    print(f"  {'━'*44}")
    print(f"  Open in browser: http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop")
    print(f"  {'━'*44}\n")

    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.\n")
