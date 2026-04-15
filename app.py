#!/usr/bin/env python3
"""
KRONOS AI FORECAST — CLOUD VERSION
===================================
Runs on Render.com free tier.
Accessible from any device including iPhone.
"""

from flask import Flask, request, jsonify, render_template_string
import json, os, requests as req
from datetime import datetime, timezone

app = Flask(__name__)

OPENAI_KEY  = os.environ.get("OPENAI_API_KEY", "")
GROK_KEY    = os.environ.get("XAI_API_KEY", "")
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
MISTRAL_KEY  = os.environ.get("MISTRAL_API_KEY", "")
CLAUDE_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
BINANCE     = "https://api.binance.com/api/v3"

MS_MAP   = {"5m":300000,"15m":900000,"30m":1800000,"1h":3600000,"4h":14400000,"1d":86400000}
IV_LABEL = {"5m":"next 2h","15m":"next 6h","30m":"next 12h","1h":"next 24h","4h":"next 4d","1d":"next 2w"}

# ── DATA ──────────────────────────────────────────────────────────────────────
def fetch_klines(symbol, interval, limit=80):
    r = req.get(f"{BINANCE}/klines",
                params={"symbol":symbol,"interval":interval,"limit":limit},timeout=15)
    r.raise_for_status()
    return [{"t":k[0],"o":float(k[1]),"h":float(k[2]),"l":float(k[3]),"c":float(k[4]),"v":float(k[5])} for k in r.json()]

def fetch_ticker(symbol):
    r = req.get(f"{BINANCE}/ticker/24hr",params={"symbol":symbol},timeout=15)
    r.raise_for_status()
    return r.json()

def time_label(ts_ms, interval):
    dt = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc)
    if interval == "1d":               return dt.strftime("%b %d")
    if interval in ("5m","15m","30m"): return dt.strftime("%H:%M")
    return dt.strftime("%m/%d %H:%M")

# ── PROMPT ────────────────────────────────────────────────────────────────────
def build_prompt(symbol, interval, klines, ticker):
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

# ── ENGINES ───────────────────────────────────────────────────────────────────
def run_gpt4o(symbol, interval, klines, ticker):
    system, user = build_prompt(symbol, interval, klines, ticker)
    r = req.post("https://api.openai.com/v1/chat/completions",
        headers={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"},
        json={"model":"gpt-4o","messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.25,"max_tokens":1500,"response_format":{"type":"json_object"}},timeout=30)
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"].strip())

def run_grok(symbol, interval, klines, ticker):
    system, user = build_prompt(symbol, interval, klines, ticker)
    r = req.post("https://api.x.ai/v1/chat/completions",
        headers={"Authorization":f"Bearer {GROK_KEY}","Content-Type":"application/json"},
        json={"model":"grok-3","messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.25,"max_tokens":1500},timeout=30)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json","").replace("```","").strip()
    start = raw.find("{"); end = raw.rfind("}")+1
    return json.loads(raw[start:end] if start>=0 and end>start else raw)

def run_gemini(symbol, interval, klines, ticker):
    system, user = build_prompt(symbol, interval, klines, ticker)
    prompt = system + "\n\n" + user

    # Try Vertex AI endpoint (works with AQ. OAuth tokens)
    project_id = os.environ.get("GOOGLE_PROJECT_ID", "sunenergy-dashboard")
    location   = "us-central1"
    for model in ["gemini-2.5-flash", "gemini-2.0-flash-001", "gemini-1.5-pro"]:
        try:
            url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/"
                   f"{project_id}/locations/{location}/publishers/google/models/"
                   f"{model}:generateContent")
            r = req.post(url,
                headers={"Authorization": f"Bearer {GEMINI_KEY}",
                         "Content-Type": "application/json"},
                json={"contents":[{"role":"user","parts":[{"text":prompt}]}],
                      "generationConfig":{"temperature":0.25,"maxOutputTokens":8192}},
                timeout=60)
            if not r.ok:
                continue
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            raw = raw.replace("```json","").replace("```","").strip()
            start = raw.find("{"); end = raw.rfind("}")+1
            return json.loads(raw[start:end] if start>=0 and end>start else raw)
        except Exception:
            continue

    # Fallback: try standard AI Studio endpoint
    for model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            r = req.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"Content-Type":"application/json","x-goog-api-key":GEMINI_KEY},
                json={"system_instruction":{"parts":[{"text":system}]},
                      "contents":[{"role":"user","parts":[{"text":user}]}],
                      "generationConfig":{"temperature":0.25,"maxOutputTokens":8192}},
                timeout=60)
            if not r.ok:
                continue
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            raw = raw.replace("```json","").replace("```","").strip()
            start = raw.find("{"); end = raw.rfind("}")+1
            return json.loads(raw[start:end] if start>=0 and end>start else raw)
        except Exception:
            continue

    raise Exception(f"Gemini failed. Key: {GEMINI_KEY[:8] if GEMINI_KEY else 'NOT SET'}")

def run_mistral(symbol, interval, klines, ticker):
    system, user = build_prompt(symbol, interval, klines, ticker)
    r = req.post("https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization":f"Bearer {MISTRAL_KEY}","Content-Type":"application/json"},
        json={"model":"mistral-large-latest",
              "messages":[{"role":"system","content":system},{"role":"user","content":user}],
              "temperature":0.25,"max_tokens":1500,
              "response_format":{"type":"json_object"}},
        timeout=30)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json","").replace("```","").strip()
    start = raw.find("{"); end = raw.rfind("}")+1
    return json.loads(raw[start:end] if start>=0 and end>start else raw)

def run_claude(symbol, interval, klines, ticker):
    system, user = build_prompt(symbol, interval, klines, ticker)
    r = req.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key":CLAUDE_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
        json={"model":"claude-haiku-4-5","max_tokens":1500,"system":system,
              "messages":[{"role":"user","content":user}]},timeout=30)
    r.raise_for_status()
    raw = r.json()["content"][0]["text"].strip()
    raw = raw.replace("```json","").replace("```","").strip()
    return json.loads(raw)

ENGINES = [
    ("GPT-4o",   run_gpt4o,   lambda: bool(OPENAI_KEY)),
    ("Grok-3",   run_grok,    lambda: bool(GROK_KEY)),
    ("Gemini",   run_gemini,  lambda: bool(GEMINI_KEY)),
    ("Mistral",  run_mistral, lambda: bool(MISTRAL_KEY)),
    ("Claude",   run_claude,  lambda: bool(CLAUDE_KEY)),
]

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Kronos AI">
<title>Kronos AI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#e2e8f0;min-height:100vh;min-height:-webkit-fill-available}
.topbar{background:#161b27;border-bottom:1px solid #21293d;padding:12px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;position:sticky;top:0;z-index:100}
.logo{font-size:17px;font-weight:700;color:#f1f5f9;white-space:nowrap}
.logo span{color:#378ADD}
select{background:#0d1117;color:#e2e8f0;border:1px solid #2d3748;border-radius:8px;padding:8px 10px;font-size:13px;cursor:pointer;outline:none;-webkit-appearance:none;flex:1;min-width:0}
.run-btn{background:#378ADD;color:#fff;border:none;border-radius:8px;padding:9px 16px;font-size:13px;font-weight:700;cursor:pointer;white-space:nowrap;-webkit-appearance:none}
.run-btn:disabled{background:#2d3748;color:#64748b}
.main{padding:14px 16px;max-width:700px;margin:0 auto}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.met{background:#161b27;border-radius:10px;padding:11px 13px;border:1px solid #21293d}
.ml{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
.mv{font-size:17px;font-weight:700;color:#f1f5f9}
.ms{font-size:11px;margin-top:2px;color:#64748b}
.up{color:#1D9E75}.dn{color:#E24B4A}
.cbox{background:#161b27;border-radius:10px;padding:14px;border:1px solid #21293d;margin-bottom:12px}
.engine-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.ecard{background:#0d1117;border-radius:10px;padding:12px;border:1px solid #21293d}
.pbar{background:#1e2d40;border-radius:4px;height:6px;overflow:hidden;margin:5px 0 3px}
.pf{height:100%;border-radius:4px;transition:width .8s ease}
.consensus{border-radius:12px;padding:16px;margin-bottom:12px}
.status{font-size:12px;color:#64748b;padding:6px 0;min-height:20px}
.abox{background:#0d1117;border-radius:8px;padding:12px;font-size:13px;line-height:1.65;color:#cbd5e1;border-left:3px solid #378ADD;margin-top:10px}
.empty{text-align:center;padding:50px 20px;color:#475569;font-size:14px}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #2d3748;border-top-color:#378ADD;border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.badge{display:inline-block;font-size:10px;font-weight:700;padding:2px 8px;border-radius:12px}
@media(min-width:600px){.metrics{grid-template-columns:repeat(4,1fr)}.engine-grid{grid-template-columns:repeat(4,1fr)}}
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">Kronos <span>AI</span></div>
  <select id="sym">
    <option value="BTCUSDT">BTC/USDT</option>
    <option value="ETHUSDT">ETH/USDT</option>
    <option value="SOLUSDT">SOL/USDT</option>
    <option value="BNBUSDT">BNB/USDT</option>
    <option value="XRPUSDT">XRP/USDT</option>
    <option value="ADAUSDT">ADA/USDT</option>
    <option value="TAOUSDT">TAO/USDT</option>
    <option value="ZECUSDT">ZEC/USDT</option>
    <option value="DOGEUSDT">DOGE/USDT</option>
    <option value="AVAXUSDT">AVAX/USDT</option>
  </select>
  <select id="iv">
    <option value="15m">15m</option>
    <option value="1h" selected>1h</option>
    <option value="4h">4h</option>
    <option value="1d">1d</option>
  </select>
  <button class="run-btn" id="runBtn" onclick="runForecast()">Run</button>
</div>

<div class="main">
  <div class="status" id="status"></div>
  <div id="content"><div class="empty">Select a symbol and tap Run</div></div>
</div>

<script>
let chart = null;
function fp(v){
  if(v>=10000) return '$'+v.toLocaleString('en-US',{maximumFractionDigits:0});
  if(v>=1) return '$'+v.toFixed(2);
  return '$'+v.toFixed(5);
}

async function runForecast(){
  const sym=document.getElementById('sym').value;
  const iv=document.getElementById('iv').value;
  const btn=document.getElementById('runBtn');
  const status=document.getElementById('status');
  btn.disabled=true;
  status.innerHTML='<span class="spin"></span>Running all engines...';
  try{
    const res=await fetch(`/forecast?symbol=${sym}&interval=${iv}`);
    const data=await res.json();
    if(data.error){status.textContent='Error: '+data.error;btn.disabled=false;return;}
    renderDashboard(data);
    status.textContent='Updated '+new Date().toLocaleTimeString();
  }catch(e){status.textContent='Error: '+e.message;}
  btn.disabled=false;
}

function renderDashboard(d){
  const cons=d.consensus;
  const price=d.price,chg=d.chg;
  const cc=chg>=0?'#1D9E75':'#E24B4A';
  const votes=cons.bullish_votes,total=cons.total;
  const cvote=votes>=Math.ceil(total/2)?'#1D9E75':votes===0?'#E24B4A':'#BA7517';
  const clabel=`${votes}/${total} ${votes>=Math.ceil(total/2)?'BULLISH':votes===0?'BEARISH':'BULLISH'}`;
  const csignal=votes>=Math.ceil(total/2)?'BUY':votes===0?'SELL':'NEUTRAL';
  const csbg=votes>=Math.ceil(total/2)?'rgba(29,158,117,0.15)':votes===0?'rgba(226,75,74,0.15)':'rgba(186,117,23,0.15)';
  const cborder=votes>=Math.ceil(total/2)?'#1D9E75':votes===0?'#E24B4A':'#BA7517';

  const closes=d.klines.map(k=>k.c);
  const times=d.klines.map(k=>k.tl);
  const fc=d.forecast_candles||[];
  const allT=[...times,...d.fc_times];
  const hist=[...closes,...Array(fc.length).fill(null)];
  const pad=Array(closes.length-1).fill(null);
  const fcmD=[...pad,closes[closes.length-1],...fc.map(c=>c.mean)];
  const fchD=[...pad,closes[closes.length-1],...fc.map(c=>c.high)];
  const fclD=[...pad,closes[closes.length-1],...fc.map(c=>c.low)];

  const ecards=d.engines.map(e=>{
    const ec=e.bullish_probability>=50?'#1D9E75':'#E24B4A';
    const et=(e.trend||'neutral').toUpperCase();
    const etc=et==='BULLISH'?'#1D9E75':et==='BEARISH'?'#E24B4A':'#888';
    return `<div class="ecard">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:12px;font-weight:600;color:#94a3b8">${e.name}</span>
        <span class="badge" style="background:${ec}22;color:${etc}">${et}</span>
      </div>
      <div class="pbar"><div class="pf" style="width:${e.bullish_probability}%;background:${ec}"></div></div>
      <div style="font-size:16px;font-weight:700;color:${ec}">${e.bullish_probability}%</div>
      <div style="font-size:11px;color:#475569;margin-top:5px">Target: <span style="color:#f1f5f9">${fp(e.target_price)}</span></div>
      <div style="font-size:11px;color:#475569">Stop: <span style="color:#E24B4A">${fp(e.stop_loss)}</span></div>
    </div>`;
  }).join('');

  document.getElementById('content').innerHTML=`
<div class="consensus" style="background:${csbg};border:1px solid ${cborder}">
  <div style="font-size:11px;color:${cborder};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Consensus — ${d.symbol} ${d.interval.toUpperCase()}</div>
  <div style="display:flex;align-items:center;justify-content:space-between">
    <div>
      <div style="font-size:26px;font-weight:700;color:${cvote}">${clabel}</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:2px">${cons.agreement} · avg ${cons.avg_bullish}% bullish</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:11px;color:#475569">Signal</div>
      <div style="font-size:20px;font-weight:700;color:${cvote}">${csignal}</div>
      <div style="font-size:10px;color:#475569">${cons.risk.toUpperCase()}</div>
    </div>
  </div>
</div>

<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:12px;flex-wrap:wrap">
  <span style="font-size:16px;font-weight:700;color:#f1f5f9">${d.symbol} · ${d.interval.toUpperCase()}</span>
  <span style="font-size:22px;font-weight:700;color:#f1f5f9">${fp(price)}</span>
  <span style="font-size:13px;color:${cc}">${chg>=0?'+':''}${chg.toFixed(2)}% 24h</span>
</div>

<div class="metrics">
  <div class="met"><div class="ml">Volume 24h</div><div class="mv" style="font-size:15px">${d.vol}</div></div>
  <div class="met"><div class="ml">24h High</div><div class="mv" style="font-size:15px">${fp(d.high24)}</div></div>
  <div class="met"><div class="ml">24h Low</div><div class="mv" style="font-size:15px">${fp(d.low24)}</div></div>
  <div class="met"><div class="ml">Avg Target</div><div class="mv" style="font-size:15px;color:#1D9E75">${fp(cons.avg_target)}</div></div>
</div>

<div class="engine-grid">${ecards}</div>

<div class="cbox">
  <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Price chart + forecast</div>
  <div style="position:relative;width:100%;height:220px">
    <canvas id="mc" role="img" aria-label="chart"></canvas>
  </div>
</div>

<div class="cbox">
  <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.5px">Combined analysis</div>
  <div class="abox">${cons.combined_analysis}</div>
</div>`;

  if(chart){chart.destroy();chart=null;}
  chart=new Chart(document.getElementById('mc').getContext('2d'),{
    type:'line',data:{labels:allT,datasets:[
      {label:'Historical',data:hist,borderColor:'#378ADD',backgroundColor:'rgba(55,138,221,0.07)',borderWidth:2,pointRadius:0,fill:true,tension:0.3},
      {label:'High',data:fchD,borderColor:'rgba(127,119,221,0.2)',backgroundColor:'rgba(127,119,221,0.08)',borderWidth:1,pointRadius:0,fill:'+1',tension:0.3},
      {label:'Low',data:fclD,borderColor:'rgba(127,119,221,0.2)',borderWidth:1,pointRadius:0,fill:false,tension:0.3},
      {label:'Forecast',data:fcmD,borderColor:'#7F77DD',borderWidth:2,borderDash:[5,3],pointRadius:0,fill:false,tension:0.3}
    ]},
    options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{backgroundColor:'#1e2330',borderColor:'#2d3748',borderWidth:1,titleColor:'#94a3b8',bodyColor:'#e2e8f0',
        callbacks:{label:c=>c.parsed.y===null?null:` ${c.dataset.label}: ${fp(c.parsed.y)}`}}},
      scales:{
        x:{ticks:{maxTicksLimit:6,color:'#475569',font:{size:10},autoSkip:true},grid:{color:'rgba(255,255,255,0.04)'}},
        y:{ticks:{color:'#475569',font:{size:10},callback:v=>v>=1000?'$'+v.toLocaleString('en-US',{maximumFractionDigits:0}):'$'+v.toFixed(2)},grid:{color:'rgba(255,255,255,0.04)'}}
      }
    }
  });
}
</script>
</body>
</html>"""

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "OPENAI_API_KEY":    "OK" if OPENAI_KEY  else "MISSING",
        "XAI_API_KEY":       "OK" if GROK_KEY    else "MISSING",
        "GEMINI_API_KEY":    "OK" if GEMINI_KEY  else "MISSING",
        "MISTRAL_API_KEY":   "OK" if MISTRAL_KEY else "MISSING",
        "ANTHROPIC_API_KEY": "OK" if CLAUDE_KEY  else "MISSING",
        "gemini_key_start":  GEMINI_KEY[:8]  if GEMINI_KEY  else "NOT SET",
        "mistral_key_start": MISTRAL_KEY[:8] if MISTRAL_KEY else "NOT SET",
    })

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/forecast")
def forecast():
    symbol   = request.args.get("symbol","BTCUSDT").upper()
    interval = request.args.get("interval","1h")

    try:
        klines = fetch_klines(symbol, interval)
        ticker = fetch_ticker(symbol)
    except Exception as e:
        return jsonify({"error":f"Binance: {e}"})

    price  = float(ticker["lastPrice"])
    vol    = float(ticker["quoteVolume"])

    def fvol(v):
        if v>=1e9: return f"{v/1e9:.1f}B"
        if v>=1e6: return f"{v/1e6:.0f}M"
        return f"{v/1e3:.0f}K"

    results, errors = [], []
    for name, fn, check in ENGINES:
        if not check():
            errors.append(f"{name}: API key not set")
            continue
        try:
            r = fn(symbol, interval, klines, ticker)
            r["name"] = name
            results.append(r)
        except Exception as e:
            errors.append(f"{name}: {e}")

    if not results:
        return jsonify({"error":"All engines failed: " + " | ".join(errors)})

    bulls   = [r["bullish_probability"] for r in results]
    trends  = [r.get("trend","neutral") for r in results]
    targets = [r.get("target_price", price) for r in results]
    stops   = [r.get("stop_loss",    price) for r in results]
    risks   = [r.get("risk_level","medium") for r in results]

    avg_bull   = round(sum(bulls)/len(bulls))
    avg_target = round(sum(targets)/len(targets), 4)
    avg_stop   = round(sum(stops)/len(stops), 4)
    bull_votes = sum(1 for b in bulls if b >= 50)
    bear_votes = sum(1 for t in trends if t == "bearish")
    bull_trend = sum(1 for t in trends if t == "bullish")
    risk_order = {"low":0,"medium":1,"high":2,"extreme":3}
    cons_risk  = max(risks, key=lambda x: risk_order.get(x,1))

    if bull_votes == len(results):     agreement = "Strong consensus — all engines bullish"
    elif bull_votes >= len(results)//2+1: agreement = "Majority bullish"
    elif bear_votes == len(results):   agreement = "Strong consensus — all engines bearish"
    elif bear_votes >= len(results)//2+1: agreement = "Majority bearish"
    else:                              agreement = "Mixed signals — engines disagree"

    combined_analysis = (
        f"{agreement}. Across {len(results)} engines ({', '.join([r['name'] for r in results])}), "
        f"average bullish probability is {avg_bull}% with {bull_trend} bullish and {bear_votes} bearish signals. "
        f"Average target: {avg_target:.4f} · Stop: {avg_stop:.4f} · Risk: {cons_risk.upper()}."
    )
    if errors:
        combined_analysis += f" Note: {'; '.join(errors)}."

    ms     = MS_MAP[interval]
    last_t = klines[-1]["t"]
    # Use first result's forecast candles
    fc     = results[0].get("forecast_candles",[])
    fc_len = len(fc)

    return jsonify({
        "symbol":   symbol,
        "interval": interval,
        "price":    price,
        "chg":      float(ticker["priceChangePercent"]),
        "vol":      fvol(vol),
        "high24":   float(ticker["highPrice"]),
        "low24":    float(ticker["lowPrice"]),
        "klines":   [{"c":k["c"],"tl":time_label(k["t"],interval)} for k in klines],
        "fc_times": [time_label(last_t+ms*(i+1),interval) for i in range(fc_len)],
        "forecast_candles": fc,
        "engines":  results,
        "consensus":{
            "bullish_votes": bull_votes,
            "total":         len(results),
            "avg_bullish":   avg_bull,
            "avg_target":    avg_target,
            "agreement":     agreement,
            "risk":          cons_risk,
            "combined_analysis": combined_analysis,
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
