# ДОПОЛНИТЕЛЬНЫЕ ПРОБЛЕМЫ В КОДЕ

Дата: 2025-12-16  
Приоритет: MEDIUM-LOW  
Статус: Найдено 7 проблем

---

## 🟡 MEDIUM PRIORITY

### 1. ❌ NaN/Infinity в индикаторах не проверяются

**Проблема:**
```python
# Строка 700-705
current_rsi = df['rsi'].iloc[-1]
current_ema20 = df['ema20'].iloc[-1]
current_macd = df['macd'].iloc[-1]
current_atr = df['atr'].iloc[-1]

# ❌ Если недостаточно данных, индикаторы = NaN!
# ❌ Деление на ноль может дать infinity
```

**Последствия:**
- RSI может быть NaN если < 14 свечей
- ATR может быть NaN
- Сравнения вроде `current_rsi < 30` вернут False для NaN
- Фильтры работают некорректно
- AI получает NaN в данных

**Решение:**
```python
import math

# Проверка на NaN/Inf
if math.isnan(current_rsi) or math.isinf(current_rsi):
    logger.warning(f"{symbol}: Invalid RSI value")
    return None

# Или используем pandas
if pd.isna([current_rsi, current_ema20, current_atr]).any():
    logger.warning(f"{symbol}: NaN in indicators")
    return None
```

---

### 2. ❌ execute_trade() НЕ вызывается при подтверждении!

**Проблема:**
```python
# В approve_trade_command() (строка 1507-1560):
if agent.paper_trading:
    agent.db.save_trade(...)  # ✅ Сохраняет в БД
    # ❌ НО НЕ добавляет в active_positions!
    
# Метод execute_trade() существует (строка 1093)
# Но НИКОГДА НЕ вызывается!
```

**Последствия:**
- Сделка сохраняется в БД
- НО НЕ добавляется в active_positions
- Мониторинг не работает для этой позиции
- Trailing stop не обновляется
- Автозакрытие не работает

**Решение:**
```python
# В approve_trade_command():
if agent.paper_trading:
    # Используем execute_trade вместо прямого save_trade
    success = agent.execute_trade(trade_id)
    if success:
        reply_text = "Сделка одобрена и добавлена в мониторинг"
```

---

### 3. ❌ Двойной расчет ATR при восстановлении позиций

**Проблема:**
```python
# В _restore_active_positions() (строка 422-454):
self.active_positions[symbol] = {
    # ...
    'atr': 0  # ❌ Всегда 0 при восстановлении!
}

# Но в update_trailing_stop():
trailing_distance = atr * 2.0  # ❌ 0 * 2.0 = 0!
```

**Последствия:**
- Восстановленные позиции имеют atr=0
- Trailing stop не работает корректно
- Расстояние trailing stop = 0

**Решение:**
```python
# При восстановлении - пересчитать ATR:
try:
    ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=50)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = df['tr'].rolling(window=14).mean().iloc[-1]
except:
    atr = entry_price * 0.02  # Fallback 2%

self.active_positions[symbol]['atr'] = atr
```

---

### 4. ❌ Асинхронный хаос в send_signal_to_telegram

**Проблема:**
```python
# Строка 994-1003
try:
    loop = asyncio.get_running_loop()
    asyncio.create_task(...)  # ❌ Таск создается но не ждет!
except RuntimeError:
    loop.run_until_complete(...)  # ✅ Корректно
```

**Последствия:**
- В первом случае (running loop) сообщение может не отправиться
- Таск "выстреливает и забывается"
- Нет гарантии доставки

**Решение:**
```python
try:
    loop = asyncio.get_running_loop()
    # Используем ensure_future вместо create_task
    future = asyncio.ensure_future(
        self.send_telegram_message_with_buttons(...)
    )
    # Можно добавить callback на ошибку
    future.add_done_callback(lambda f: logger.error(f"Send failed: {f.exception()}") if f.exception() else None)
except RuntimeError:
    ...
```

---

## 🟢 LOW PRIORITY (улучшения)

### 5. ⚠️ Нет сохранения ATR в БД

**Проблема:**
```python
# В database.py таблица trades НЕ имеет поля atr
# При save_trade() atr не сохраняется
# При восстановлении - atr=0
```

**Решение:**
- Добавить колонку `atr REAL` в таблицу trades
- Сохранять при create
- Восстанавливать при load

---

### 6. ⚠️ Время в БД может быть строкой или datetime

**Проблема:**
```python
# Строка 442-443
'entry_time': datetime.fromisoformat(trade['entry_time']) 
              if isinstance(trade['entry_time'], str) 
              else trade['entry_time']
```

**Почему это проблема:**
- SQLite хранит datetime как TEXT
- При get_open_trades() возвращается строка
- Нужна конвертация каждый раз

**Решение:**
```python
# В database.py - всегда возвращать datetime:
def get_open_trades(self):
    trades = [dict(row) for row in rows]
    for trade in trades:
        if isinstance(trade['entry_time'], str):
            trade['entry_time'] = datetime.fromisoformat(trade['entry_time'])
    return trades
```

---

### 7. ⚠️ Hardcoded операторский chat_id

**Проблема:**
```python
# В нескольких местах:
operator_chat_id = "5150355926"  # ❌ Hardcoded!
```

**Решение:**
```python
# В .env:
OPERATOR_CHAT_ID=5150355926

# В коде:
self.operator_chat_id = os.getenv('OPERATOR_CHAT_ID')
```

---

## 📊 ПРИОРИТЕТЫ ИСПРАВЛЕНИЙ

### КРИТИЧНО (сделать сейчас):
1. ✅ **#2: execute_trade() не вызывается** - позиции не мониторятся!

### HIGH:
2. **#1: NaN в индикаторах** - может ломать сигналы
3. **#3: ATR=0 при восстановлении** - trailing stop не работает

### MEDIUM:
4. **#4: Асинхронный хаос** - потеря сообщений

### LOW:
5. #5: ATR в БД
6. #6: Типы datetime
7. #7: Hardcoded chat_id

---

## 🎯 ИТОГО

**Критических:** 1 (execute_trade не вызывается)  
**Высоких:** 2 (NaN, ATR=0)  
**Средних:** 1 (async chaos)  
**Низких:** 3 (улучшения)

Самая важная проблема #2 - позиции одобряются но НЕ добавляются в мониторинг!
