# Подключение Supabase к боту

## ✅ Что сделано:

1. **Добавлены ключи в .env**
   - SUPABASE_URL=https://ixovpisndoyhsaaqlypl.supabase.co
   - SUPABASE_KEY (анонимный ключ)
   - SUPABASE_SERVICE_KEY (сервисный ключ)

2. **Установлен Python клиент**
   ```bash
   pip install supabase>=2.26.0
   ```

3. **Создан модуль database_supabase.py**
   - Класс `SupabaseDatabase` для работы с облачной БД
   - Методы: save_signal, save_trade, update_trade, get_statistics

## 🔧 Следующие шаги:

### 1. Создать таблицы в Supabase

Открой: https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl/sql

Скопируй и выполни SQL из файла `supabase_setup.sql`:

```sql
-- Таблицы: signals, trades, performance
-- С индексами и политиками безопасности
```

### 2. Протестировать подключение

```bash
python test_supabase.py
```

Скрипт создаст тестовый сигнал и сделку, проверит все функции.

### 3. Интегрировать в trading_bot.py

В файле trading_bot.py добавить:

```python
from database_supabase import SupabaseDatabase

# В __init__
self.supabase_db = SupabaseDatabase()

# При создании сигнала (дополнительно к SQLite)
self.supabase_db.save_signal(...)

# При открытии сделки
self.supabase_db.save_trade(...)

# При закрытии сделки
self.supabase_db.update_trade(...)
```

## 🎯 Преимущества Supabase:

- ☁️ Облачное хранение (доступно везде)
- 🔄 Автоматический бэкап
- 📊 Веб-интерфейс для аналитики
- 🌐 Real-time обновления
- 🔒 Row Level Security
- 📈 Масштабируемость

## 📊 Просмотр данных:

**Table Editor:**
https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl/editor

**SQL Editor:**
https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl/sql

## 🔐 Безопасность:

- Ключи уже добавлены в .env
- .env не загружается в GitHub (.gitignore)
- Row Level Security включена
- Только авторизованные запросы

## 🚀 Готово к использованию!

После создания таблиц в SQL Editor, бот будет автоматически сохранять все сигналы и сделки как в локальную SQLite, так и в облачную Supabase.
