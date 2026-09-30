# TLT / SUOA signal monitor

Щоденний моніторинг двох треків, що рухають пару TLT/SUOA, зі сповіщенням у Telegram при зміні стану.

| Трек | Джерела | Індикатори |
|---|---|---|
| **Ставки** | FRED: DGS2/10/30, Yahoo: TLT, SUOA | Ratio TLT/SUOA: SMA50/200, перетин, MACD, RSI + дивергенції, пробій 6-міс. діапазону; тренд 10Y; нахил 2s10s і 10s30s; реалізована вол. 10Y (заміна MOVE) |
| **Кредит** | FRED: ICE BofA IG OAS, BBB OAS | Тренд OAS за 20д, OAS vs SMA50, режим SMA50/200, декомпресія BBB−IG, прискорення, z-score за рік |
| **Макро** | FRED: CPI, NFP, UNRATE | Інфляція (YoY vs 3 міс. тому), ринок праці — контекст, у скоринг не входить |

**Скоринг:** кожен індикатор голосує +1 (на користь TLT) / −1 (на користь SUOA) / 0.
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
5. **Перевірте тікер SUOA** у `config.yaml` (за замовчуванням `SUOA.L`) на finance.yahoo.com —
   клас фонду має збігатися з тим, який ви тримаєте.
6. Actions → `signals` → **Run workflow**. Перший запуск надішле повне зведення в Telegram,
   далі — лише при змінах (або щодня, якщо `telegram.daily_summary: true`).

## Локально

```bash
pip install -r requirements.txt
python signals.py --demo --no-notify      # синтетичні дані, перевірка логіки
FRED_API_KEY=... python signals.py --no-notify
```

## Тюнінг

Усі пороги — у `config.yaml` (вікна SMA, пороги зміни в б.п., вікно діапазону, пороги волатильності).
Якщо алертів забагато — збільште `*_chg_bp` або `range_skip`.

## Застереження

- TLT (NYSE) і SUOA (LSE) мають різні торгові дні — ratio рахується лише по спільних датах.
  Використовуються скориговані ціни (з дивідендами).
- Ratio TLT/SUOA здебільшого відображає різницю дюрацій (≈17 vs ≈6–7), тому трек «Кредит»
  рахується окремо з OAS — саме він показує кредитний ризик.
- Це інструмент моніторингу, не інвестиційна порада.
