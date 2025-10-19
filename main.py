import os
import time
import logging
from typing import List, Optional, Dict, Any, Tuple

import requests
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", level=logging.INFO)
log = logging.getLogger("mudrex_mi_bot")

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

BINANCE_HOSTS = ["https://api.binance.com", "https://api-gcp.binance.com", "https://api1.binance.com"]
HTTP_TIMEOUT = 5
EXCHANGEINFO_TTL = 1800

_exchangeinfo_cache: Dict[str, Any] = {"expires": 0, "data": None}
PREFERRED_QUOTES = ["USDT", "FDUSD", "USDC"]
VALID_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}
DEFAULT_INTERVAL = "1h"

def _http_get(path: str, params: dict) -> Optional[requests.Response]:
    last_exc = None
    for host in BINANCE_HOSTS:
        url = f"{host}{path}"
        for attempt in range(2):
            try:
                r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if r.status_code == 200:
                    return r
            except requests.RequestException as e:
                last_exc = e
                time.sleep(0.3)
    log.error(f"HTTP GET failed: {path} - {last_exc}")
    return None

def _get_exchangeinfo() -> Optional[dict]:
    now = time.time()
    if _exchangeinfo_cache["data"] and now < _exchangeinfo_cache["expires"]:
        return _exchangeinfo_cache["data"]
    r = _http_get("/api/v3/exchangeInfo", params={})
    if not r or r.status_code != 200:
        return None
    try:
        data = r.json()
        _exchangeinfo_cache["data"] = data
        _exchangeinfo_cache["expires"] = now + EXCHANGEINFO_TTL
        return data
    except Exception as e:
        log.error(f"Failed to parse exchangeInfo: {e}")
        return None

def _symbol_exists(symbol: str) -> bool:
    info = _get_exchangeinfo()
    if info and "symbols" in info:
        if symbol in {s["symbol"] for s in info["symbols"]}:
            return True
    r = _http_get("/api/v3/exchangeInfo", params={"symbol": symbol})
    return bool(r and r.status_code == 200)

def _normalize_command(text: str) -> Tuple[str, Optional[str]]:
    s = text.strip().lstrip("/").split("@")[0]
    parts = s.split()
    symbol = parts[0].upper().replace("/", "")
    tf = parts[1].lower() if len(parts) > 1 else None
    return symbol, tf

def _resolve_symbol_or_fallback(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    if _symbol_exists(symbol):
        return symbol, None
    for q in sorted(PREFERRED_QUOTES, key=len, reverse=True):
        if symbol.endswith(q):
            base = symbol[:-len(q)]
            if not base:
                break
            for alt in PREFERRED_QUOTES:
                alt_sym = f"{base}{alt}"
                if _symbol_exists(alt_sym):
                    return alt_sym, f"Resolved to {alt_sym}"
            break
    for q in PREFERRED_QUOTES:
        candidate = f"{symbol}{q}"
        if _symbol_exists(candidate):
            return candidate, None
    return None, None

def fetch_ticker_24h(symbol: str) -> Optional[dict]:
    r = _http_get("/api/v3/ticker/24hr", params={"symbol": symbol})
    if not r or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception as e:
        log.error(f"Failed to parse ticker: {e}")
        return None

def fetch_klines(symbol: str, interval: str, limit: int = 100) -> Optional[List[List]]:
    r = _http_get("/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
    if not r or r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception as e:
        log.error(f"Failed to parse klines: {e}")
        return None

class TechnicalAnalysis:
    def __init__(self, closes: List[float]):
        if len(closes) < 30:
            raise ValueError("Need at least 30 closing prices")
        self.prices = np.array(closes, dtype=float)

    def _ema(self,  np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def calculate_rsi(self, period: int = 14) -> float:
        delta = np.diff(self.prices)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    def calculate_macd(self, short: int = 12, long: int = 26, signal: int = 9):
        macd = self._ema(self.prices, short) - self._ema(self.prices, long)
        signal_line = self._ema(macd, signal)
        macd_val = round(macd[-1], 4)
        signal_val = round(signal_line[-1], 4)
        trend = "Bullish" if macd_val > signal_val else "Bearish"
        return macd_val, trend

    def calculate_sma(self, period: int) -> float:
        return round(float(np.mean(self.prices[-period:])), 4)

    def get_moving_average_signal(self) -> str:
        sma7, sma25 = self.calculate_sma(7), self.calculate_sma(25)
        if sma7 > sma25 * 1.02:
            return "Strong Uptrend"
        elif sma7 > sma25:
            return "Mild Uptrend"
        elif sma7 < sma25 * 0.98:
            return "Strong Downtrend"
        elif sma7 < sma25:
            return "Mild Downtrend"
        return "Neutral"

    def rating(self) -> str:
        rsi = self.calculate_rsi()
        _, macd_trend = self.calculate_macd()
        ma_signal = self.get_moving_average_signal()
        bullish = bearish = 0
        if rsi > 70:
            bearish += 1
        elif rsi < 30:
            bullish += 1
        bullish += 1 if macd_trend == "Bullish" else 0
        bearish += 1 if macd_trend == "Bearish" else 0
        if "Uptrend" in ma_signal:
            bullish += 1
        elif "Downtrend" in ma_signal:
            bearish += 1
        if bullish == 3:
            return "Strong Buy 🟢"
        elif bullish == 2:
            return "Buy 🟢"
        elif bearish == 3:
            return "Strong Sell 🔴"
        elif bearish == 2:
            return "Sell 🔴"
        return "Neutral ⚪️"

def fmt_price(v: float) -> str:
    if v >= 1000:
        return f"${v:,.0f}"
    elif v >= 1:
        return f"${v:,.2f}"
    return f"${v:,.6f}"

def build_update(symbol_in: str, interval_in: Optional[str]) -> str:
    symbol = symbol_in.upper().replace("/", "")
    interval = (interval_in or DEFAULT_INTERVAL).lower()
    if interval not in VALID_INTERVALS:
        interval = DEFAULT_INTERVAL
    resolved, note = _resolve_symbol_or_fallback(symbol)
    if not resolved:
        return f"⚠️ Symbol `{symbol}` not found on Binance.\nTry `/BTCUSDT`, `/ETHUSDT`, or `/SOLUSDT`."
    t = fetch_ticker_24h(resolved)
    if not t or "lastPrice" not in t:
        return f"⚠️ Could not fetch 24h stats for `{resolved}`. Try again later."
    kl = fetch_klines(resolved, interval=interval, limit=100)
    ta_lines = ""
    if kl and len(kl) >= 30:
        closes = [float(row[4]) for row in kl]
        try:
            ta = TechnicalAnalysis(closes)
            ta_rating = ta.rating()
            sentiment = "Bullish 🚀" if "Buy" in ta_rating else "Bearish 🔻" if "Sell" in ta_rating else "Neutral ⚪️"
            ta_lines = f"- Market Sentiment: {sentiment}\n- Market Trend (TA): {ta_rating}\n"
        except Exception as e:
            log.warning(f"TA failed for {resolved}: {e}")
    last_price = float(t.get("lastPrice", 0))
    pct = float(t.get("priceChangePercent", 0.0))
    high = float(t.get("highPrice", last_price))
    low = float(t.get("lowPrice", last_price))
    arrow = "▲" if pct >= 0 else "▼"
    header = f"🔸 {resolved} — Market Update"
    if note:
        header += f"\n(Your input `{symbol}` {note})"
    return f"{header}\n——————————————\n{ta_lines}- Last Price: {fmt_price(last_price)}\n- 24h Change: {arrow} {abs(pct):.2f}%\n- Day High: {fmt_price(high)}\n- Day Low: {fmt_price(low)}\n——————————————\nPowered by [Mudrex Market Intelligence](https://mudrex.go.link/f8PJF)"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Welcome to Mudrex MI Bot!\nSend `/BTCUSDT`, `/ETHUSDT`, etc. Optionally add timeframe: `/BTCUSDT 15m`.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Usage:\n• `/BTCUSDT` – default 1h timeframe\n• `/BTCUSDT 15m` – specify timeframe\nSupported: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w")

async def any_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    text = update.message.text or ""
    symbol, tf = _normalize_command(text)
    msg = build_update(symbol, tf)
    await update.message.reply_text(msg, parse_mode="Markdown", disable_web_page_preview=True)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception(f"Unhandled error: {context.error}")

def main():
    log.info("=" * 60)
    log.info("MUDREX MARKET INTELLIGENCE BOT")
    log.info("=" * 60)
    if not TELEGRAM_TOKEN:
        log.error("❌ TELEGRAM_TOKEN not set!")
        return
    log.info(f"✅ Token loaded: {TELEGRAM_TOKEN[:10]}...")
    try:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_cmd))
        app.add_handler(MessageHandler(filters.COMMAND, any_command))
        app.add_error_handler(on_error)
        log.info("✅ Bot handlers registered")
        log.info("✅ Starting polling mode...")
        log.info("✅ Bot is running! Try /BTCUSDT in Telegram")
        log.info("=" * 60)
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        log.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
