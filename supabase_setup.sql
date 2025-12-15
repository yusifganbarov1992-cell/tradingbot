-- SQL скрипт для создания таблиц в Supabase
-- Выполни этот код в SQL Editor на https://supabase.com/dashboard/project/ixovpisndoyhsaaqlypl/sql

-- Таблица сигналов
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    trade_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    rsi DECIMAL(10, 2),
    ema20 DECIMAL(20, 8),
    ema50 DECIMAL(20, 8),
    macd DECIMAL(20, 8),
    volume DECIMAL(20, 8),
    avg_volume DECIMAL(20, 8),
    atr DECIMAL(20, 8),
    filters_passed INTEGER,
    ai_signal TEXT,
    ai_confidence INTEGER,
    ai_reason TEXT,
    status TEXT DEFAULT 'pending',
    amount DECIMAL(20, 8),
    usdt_amount DECIMAL(20, 8),
    fee DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Таблица сделок
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    amount DECIMAL(20, 8) NOT NULL,
    usdt_amount DECIMAL(20, 8) NOT NULL,
    fee DECIMAL(20, 8),
    entry_time TIMESTAMPTZ DEFAULT NOW(),
    exit_price DECIMAL(20, 8),
    exit_time TIMESTAMPTZ,
    pnl DECIMAL(20, 8),
    pnl_percent DECIMAL(10, 2),
    status TEXT DEFAULT 'open',
    mode TEXT DEFAULT 'paper',
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Таблица показателей производительности
CREATE TABLE IF NOT EXISTS performance (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate DECIMAL(10, 2),
    total_pnl DECIMAL(20, 8),
    avg_pnl DECIMAL(20, 8),
    max_drawdown DECIMAL(10, 2),
    balance DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance(timestamp DESC);

-- Row Level Security (RLS) - ОТКЛЮЧЕНО для service_role
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance ENABLE ROW LEVEL SECURITY;

-- Политики доступа (разрешить все операции)
DROP POLICY IF EXISTS "Allow all for authenticated users" ON signals;
DROP POLICY IF EXISTS "Allow all for authenticated users" ON trades;
DROP POLICY IF EXISTS "Allow all for authenticated users" ON performance;

CREATE POLICY "Allow all operations" ON signals
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations" ON trades
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow all operations" ON performance
    FOR ALL USING (true) WITH CHECK (true);
