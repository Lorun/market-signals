# TLT / SUOA signal monitor

Щоденний моніторинг трьох треків, що рухають пару TLT/SUOA та XUHA, зі сповіщенням у Telegram при зміні стану.

| Трек | Джерела | Індикатори |
|---|---|---|
| **Ставки** | FRED: DGS2/10/30, Yahoo: TLT, SUOA | Ratio TLT/SUOA: SMA50/200, перетин, MACD, RSI + дивергенції, пробій 6-міс. діапазону; тренд 10Y; нахил 2s10s і 10s30s; реалізована вол. 10Y (заміна MOVE) |
| **Кредит** | FRED: ICE BofA IG OAS, BBB OAS | Тренд OAS за 20д, OAS vs SMA50, режим SMA50/200, декомпресія BBB−IG, прискорення, z-score за рік |
| **Макро** | FRED: CPI, NFP, UNRATE | Інфляція (YoY vs 3 міс. тому), ринок праці — контекст, у скоринг не входить |
| **Висока дохідність** | FRED: HY OAS, BB OAS, CCC OAS, HY Yield, VIX, NFCI, SLOOS, WTI; Yahoo: XUHA.L, SUOA | XUHA: SMA50/200, MACD, RSI дивергенція, пробій діапазону; HY/IG ratio vs SMA200; тренд і режим HY OAS; декомпресія CCC−BB; VIX перцентиль; NFCI фінансові умови |

**Скоринг Ставки/Кредит:** кожен індикатор голосує +1 (на користь TLT) / −1 (на користь SUOA) / 0.

**⚠️ Скоринг Висока дохідність — ПРОТИЛЕЖНА конвенція:**
+1 = сприятливо для XUHA (risk-on), −1 = несприятливо. Не порівнюйте знаки напряму з треками Ставки/Кредит.

Алерт надсилається, коли змінюється стан будь-якого індикатора або режим скорингу трека.

## Як це працює

```
GitHub Actions (cron, Пн–Пт 22:30 UTC)
  └─ signals.py: FRED API + yfinance → індикатори → data/signals.json, data/history.csv
       ├─ зміна стану? → Telegram
       └─ commit у репо
Claude scheduled task (щотижня)
  └─ читає data/*.json|csv → аналітичний бриф
```
Уся математика — у скрипті, детермінована. Claude лише інтерпретує готові числа.

## Налаштування (~15 хв)

1. **FRED API key** — безкоштовно: https://fred.stlouisfed.org/docs/api/api_key.html
2. **Telegram-бот**
   - У Telegram напишіть `@BotFather` → `/newbot` → збережіть токен.
   - Напишіть своєму боту будь-що, потім відкрийте
     `https://api.telegram.org/bot<TOKEN>/getUpdates` і візьміть `message.chat.id`.
3. **Репозиторій**: створіть (можна приватний) і запуште ці файли.
4. **Secrets** (Settings → Secrets and variables → Actions):
   `FRED_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
5. **Перевірте тікери** у `config.yaml` на finance.yahoo.com:
   - `SUOA.SW` — USD-клас фонду (якщо інший клас — замініть).
   - `XUHA.L` — Xtrackers USD High Yield Corporate Bond UCITS ETF 1C (LSE, USD accumulating).
6. Actions → `signals` → **Run workflow**. Перший запуск надішле повне зведення в Telegram,
   далі — лише при змінах (або щодня, якщо `telegram.daily_summary: true`).

## Локально

```bash
pip install -r requirements.txt
python signals.py --demo --no-notify      # синтетичні дані, перевірка логіки
FRED_API_KEY=... python signals.py --no-notify
```

## Структура коду

```
signals.py          — оркестрація (завантаження, алерти, Telegram, запис файлів)
indicators.py       — чисті математичні функції (RSI, MACD, z-score тощо)
sources.py          — завантаження даних (FRED, yfinance, demo_data)
tracks/
  rates.py          — трек Ставки
  credit.py         — трек Кредит
  macro.py          — трек Макро + label()
  high_yield.py     — трек Висока дохідність (XUHA) + hy_label()
```

## Тюнінг

Усі пороги — у `config.yaml` (вікна SMA, пороги зміни в б.п., вікно діапазону, пороги волатильності).
Якщо алертів забагато — збільште `*_chg_bp` або `range_skip`.

## Застереження

- TLT (NYSE) і SUOA (LSE) мають різні торгові дні — ratio рахується лише по спільних датах.
  Використовуються скориговані ціни (з дивідендами).
- Ratio TLT/SUOA здебільшого відображає різницю дюрацій (≈17 vs ≈6–7), тому трек «Кредит»
  рахується окремо з OAS — саме він показує кредитний ризик.
- HY-облігації поводяться ближче до акцій, ніж до трежеріз. Трек XUHA і трек Кредит (IG)
  можуть тимчасово суперечити — це нормально в перехідних фазах кредитного циклу.
- Це інструмент моніторингу, не інвестиційна порада.
