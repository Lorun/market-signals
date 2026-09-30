# Задача: додати трек «High Yield» (XUHA) у market-signals

## Контекст
Репозиторій `market-signals` щодня (GitHub Actions) рахує сигнали для пари TLT/SUOA:
`signals.py` → `data/signals.json` + `data/history.csv` → Telegram при зміні стану.
Раз на тиждень Claude читає `data/*` і пише бриф за інструкцією в `weekly_brief_prompt.md`.

Треба додати **окремий аналітичний трек для XUHA** — Xtrackers USD High Yield Corporate Bond
UCITS ETF 1C (USD, акумулюючий, Yahoo: `XUHA.L`). HY-облігації поводяться ближче до акцій,
ніж до трежеріз: головні драйвери — HY-спред, апетит до ризику, фінансові умови та кредитний цикл.

## Спершу прочитай
`signals.py`, `config.yaml`, `README.md`, `weekly_brief_prompt.md`, `.github/workflows/signals.yml`.
Дотримуйся існуючих конвенцій (стани-рядки, голоси ±1, `label()`, `diff_states()`, `--demo`).

## Рефакторинг (перед додаванням треку)
`signals.py` уже великий. Розбий без зміни поведінки:
- `indicators.py` — `rsi`, `macd_hist`, `divergence`, `zscore`, `pct_rank`, `trend3`, `r`, + нові хелпери
- `sources.py` — `fetch_fred`, `fetch_prices`, `demo_data`
- `tracks/rates.py`, `tracks/credit.py`, `tracks/macro.py`, `tracks/high_yield.py`
- `signals.py` — лише оркестрація, алерти, Telegram, запис файлів

Перевірка: `python signals.py --demo --no-notify` до і після рефакторингу дає **ідентичний** `signals.json`
для існуючих треків (порівняй, ігноруючи `generated_utc`).

## Нові джерела (усі безкоштовні)
Додай у `config.yaml`. На початку роботи **перевір кожен FRED ID** запитом до API
(потрібен `FRED_API_KEY` у env); якщо серія не існує або не оновлюється — повідом і запропонуй заміну.

| Ключ | Джерело | Що це |
|---|---|---|
| `tickers.hy` | Yahoo `XUHA.L` | ціна XUHA (перевір, що це саме клас 1C USD) |
| `hy_oas` | FRED `BAMLH0A0HYM2` | ICE BofA US High Yield OAS |
| `hy_yield` | FRED `BAMLH0A0HYM2EY` | HY effective yield (контекст carry) |
| `hy_bb` | FRED `BAMLH0A1HYBB` | BB OAS |
| `hy_ccc` | FRED `BAMLH0A3HYC` | CCC & lower OAS |
| `vix` | FRED `VIXCLS` | VIX |
| `nfci` | FRED `NFCI` | Chicago Fed financial conditions (тижневий; > 0 = жорсткіші за середні) |
| `sloos` | FRED `DRTSCILM` | % банків, що посилюють стандарти C&I-кредитування (квартальний) |
| `oil` | FRED `DCOILWTICO` | WTI (енергетика — велика частка HY) |

IG OAS і ціна SUOA вже завантажуються — використай повторно.

## Трек `high_yield`
**Конвенція знаку: `+` = сприятливо для XUHA (risk-on), `−` = несприятливо.**
Це інша конвенція, ніж у треках TLT/SUOA — явно пропиши її в `signals.json` (`hy_convention`)
і в README.

### Стани з голосами
| Індикатор | Стани | Голос |
|---|---|---|
| `price_vs_sma200` (XUHA) | above / below | +1 / −1 |
| `sma_cross` (XUHA 50/200) | golden / death | +1 / −1 |
| `macd` (XUHA) | positive / negative | +1 / −1 |
| `rsi_divergence` (XUHA) | bullish / bearish / none | +1 / −1 / 0 |
| `hy_vs_ig_ratio` (XUHA/SUOA vs SMA200) | above / below | +1 / −1 |
| `hy_oas_trend` (зміна за `chg_window`, поріг `hy_oas_chg_bp`, деф. 25) | tightening / widening / flat | +1 / −1 / 0 |
| `hy_oas_regime` (SMA50 vs SMA200 OAS) | tightening_regime / widening_regime | +1 / −1 |
| `ccc_bb_decompression` (зміна CCC−BB, поріг деф. 25 б.п.) | compressing / decompressing / flat | +1 / −1 / 0 |
| `vix_regime` (перцентиль за рік) | low (≤30) / elevated (≥80) / normal | +1 / −1 / 0 |
| `financial_conditions` (NFCI: рівень і зміна за 4 тижні) | easing / tightening / stable | +1 / −1 / 0 |

### Стани без голосів (контекст для брифу)
- `drawdown` — просадка XUHA від 52-тижневого максимуму: `none` (<2%) / `mild` (2–5%) / `significant` (>5%)
- `hy_oas_extreme` — z-score за рік: wide / tight / normal
- `lending_standards` — SLOOS останнє значення vs попереднє: tightening / easing / stable
- `oil_trend` — зміна WTI за 20 днів, поріг ±10%: rising / falling / flat
- `range_breakout` для XUHA — як у треку rates

### Метрики
Рівні та зміни: ціна XUHA, SMA50/200, RSI, просадка %, XUHA/SUOA ratio, HY OAS (б.п.) + зміна
+ z + перцентиль, BB, CCC, CCC−BB, HY−IG gap, HY effective yield, VIX + перцентиль, NFCI, SLOOS,
WTI + зміна %, дати останніх спостережень для місячних/тижневих/квартальних серій.

## Інтеграція
- `signals.json`: новий блок `high_yield` (score, max_score, label, states, metrics).
  Для `label()` HY-треку — окремі формулювання («сприятливо / несприятливо для XUHA»).
- `diff_states()`: включи `high_yield` в алерти та `alert_log`.
- `history.csv`: додай колонки `xuha`, `hy_oas_bp`, `ccc_bb_bp`, `vix`, `hy_score`.
  Старі рядки без цих колонок мають читатися коректно (NaN).
- Telegram: окремий блок **High Yield (XUHA)** — score/label, ціна vs SMA200, HY OAS зі зміною,
  CCC−BB, VIX, NFCI.
- Перевірка застарілості (`warnings`): XUHA і HY OAS; для тижневих/квартальних серій
  використай окремі пороги (NFCI 14 днів, SLOOS 120 днів).
- `demo_data()`: згенеруй правдоподібні синтетичні ряди для всіх нових серій.
- `weekly_brief_prompt.md`: додай розділ **High Yield (XUHA)** — score і зміна за тиждень,
  стан спредів (HY, CCC−BB), VIX/NFCI, просадка, кредитний цикл (SLOOS), рівні, на які дивитися.
  Нагадай у промпті про іншу конвенцію знаку.
- `README.md`: опиши трек, джерела, конвенцію знаку.
- Workflow не змінюй (нових secrets не потрібно).

## Критерії готовності
1. `python signals.py --demo --no-notify` відпрацьовує; у виводі є блок HY.
2. Регресія: існуючі треки дають той самий результат, що й до рефакторингу.
3. Алерт спрацьовує: зміни вручну один стан HY у `data/signals.json`, перезапусти demo — алерт є.
4. Реальний прогін з `FRED_API_KEY`: усі серії завантажились, `warnings` порожній або пояснений.
5. Жодних нових платних джерел і жодних секретів у коді.
6. Короткий підсумок: що змінено, які FRED-серії перевірено, що не вдалося і чому.
