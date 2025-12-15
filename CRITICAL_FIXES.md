# КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ

## ❌ Проблема 1: Trailing Stop не работает!

### Причина:
Метод `update_trailing_stop()` ВЫЗЫВАЕТСЯ, но НЕ СУЩЕСТВУЕТ!

### Решение:
Добавить метод в класс `TradingAgent`:

```python
def update_trailing_stop(self, symbol: str, current_price: float) -> float | None:
    """Update trailing stop loss
    
    Returns:
        New stop loss price or None
    """
    if symbol not in self.active_positions:
        return None
    
    position = self.active_positions[symbol]
    side = position['side']
    entry_price = position['entry_price']
    atr = position.get('atr', 0)
    
    # ATR-based trailing stop (2x ATR)
    trailing_distance = 2 * atr if atr > 0 else entry_price * 0.02  # 2% fallback
    
    if side == 'BUY':
        # LONG: Stop должен следовать ЗА ценой ВВЕРХ
        new_stop = current_price - trailing_distance
        old_stop = position.get('stop_loss', 0)
        
        # Обновляем только если новый стоп ВЫШЕ старого
        if new_stop > old_stop:
            logger.info(f"Trailing stop updated for {symbol}: ${old_stop:.2f} -> ${new_stop:.2f}")
            return new_stop
    
    else:  # SELL (SHORT)
        # SHORT: Stop должен следовать ЗА ценой ВНИЗ
        new_stop = current_price + trailing_distance
        old_stop = position.get('stop_loss', float('inf'))
        
        # Обновляем только если новый стоп НИЖЕ старого
        if new_stop < old_stop:
            logger.info(f"Trailing stop updated for {symbol}: ${old_stop:.2f} -> ${new_stop:.2f}")
            return new_stop
    
    return None
```

---

## ❌ Проблема 2: Trailing Stop не сохраняется в БД

### Причина:
`position['stop_loss'] = new_stop` - обновляется только в памяти (active_positions)

### Решение:
Добавить в database.py метод `update_stop_loss()`:

```python
def update_stop_loss(self, trade_id, new_stop_loss):
    """Update stop loss for open trade"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE trades 
        SET stop_loss = ?
        WHERE trade_id = ? AND status = 'open'
    ''', (new_stop_loss, trade_id))
    conn.commit()
    conn.close()
```

И вызывать после обновления:
```python
new_stop = self.update_trailing_stop(symbol, current_price)
if new_stop:
    stop_loss = new_stop
    position['stop_loss'] = new_stop
    # ✅ ДОБАВИТЬ:
    self.db.update_stop_loss(position['trade_id'], new_stop)
```

---

## ❌ Проблема 3: Асинхронные сообщения теряются

### Причина:
```python
asyncio.create_task(self.send_telegram_message(...))
# Таск создается, но НЕ ЖДЕТ завершения!
```

### Решение:
Использовать `asyncio.ensure_future()` + хранить ссылку:

```python
try:
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(
        self.send_telegram_message_with_buttons(operator_chat_id, message, reply_markup)
    )
    # Не ждем завершения здесь (чтобы не блокировать), но таск выполнится
except RuntimeError:
    # Нет event loop - создаем и ждем
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        self.send_telegram_message_with_buttons(operator_chat_id, message, reply_markup)
    )
    loop.close()
```

---

## ⚠️ Проблема 4: Нет защиты от дневных потерь

### Решение:
Добавить в `TradingAgent.__init__()`:

```python
self.daily_stats = {
    'date': datetime.now().date(),
    'trades': 0,
    'pnl': 0.0,
    'max_daily_loss': -100.0  # $100 максимальный дневной убыток
}
```

Проверять перед новой сделкой:
```python
def can_trade_today(self) -> bool:
    """Check if trading allowed today"""
    today = datetime.now().date()
    
    # Reset stats if new day
    if self.daily_stats['date'] != today:
        self.daily_stats = {
            'date': today,
            'trades': 0,
            'pnl': 0.0,
            'max_daily_loss': -100.0
        }
    
    # Check daily loss limit
    if self.daily_stats['pnl'] < self.daily_stats['max_daily_loss']:
        logger.warning(f"Daily loss limit reached: ${self.daily_stats['pnl']:.2f}")
        return False
    
    return True
```

---

## 📊 ПРИОРИТЕТЫ:

1. **CRITICAL** - Добавить `update_trailing_stop()` (БЕЗ ЭТОГО trailing stop НЕ РАБОТАЕТ!)
2. **HIGH** - Сохранение trailing stop в БД (иначе теряется при рестарте)
3. **MEDIUM** - Фикс async сообщений (сейчас работает, но ненадежно)
4. **LOW** - Дневной лимит убытков (хорошая защита, но не критично)

---

## ✅ ЧТО УЖЕ РАБОТАЕТ:

- AI кэш с автоочисткой ✅
- SHORT логика закрытия ✅
- Telegram уведомления ✅
- Обработчики кнопок ✅
- Лимит 5 позиций ✅
- Валидация AI ответов ✅

## 🚀 ПОСЛЕ ИСПРАВЛЕНИЙ:

Бот будет ПОЛНОСТЬЮ автоматизирован:
- Открывает TOP-3 AI сигналы (по кнопкам) ✅
- Обновляет trailing stop каждую минуту ⏳ (нужен метод!)
- Закрывает позиции автоматически ✅
- Сохраняет всё в БД ✅
- Не теряет данные при рестарте ⏳ (нужен update_stop_loss!)
