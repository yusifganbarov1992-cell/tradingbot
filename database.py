"""
База данных для хранения истории сделок и показателей
"""
import sqlite3
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TradingDatabase:
    def __init__(self, db_path='trading_history.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Создать таблицы если не существуют"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица сигналов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                rsi REAL,
                ema20 REAL,
                ema50 REAL,
                macd REAL,
                volume REAL,
                avg_volume REAL,
                atr REAL,
                filters_passed INTEGER,
                ai_signal TEXT,
                ai_confidence INTEGER,
                ai_reason TEXT,
                status TEXT DEFAULT 'pending',
                amount REAL,
                usdt_amount REAL,
                fee REAL
            )
        ''')
        
        # Таблица выполненных сделок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                amount REAL NOT NULL,
                usdt_amount REAL NOT NULL,
                fee REAL,
                entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                exit_price REAL,
                exit_time DATETIME,
                pnl REAL,
                pnl_percent REAL,
                status TEXT DEFAULT 'open',
                mode TEXT DEFAULT 'paper',
                stop_loss REAL,
                take_profit REAL
            )
        ''')
        
        # Таблица показателей производительности
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                avg_pnl REAL,
                max_drawdown REAL,
                balance REAL
            )
        ''')
        
        # Таблица состояния системы безопасности
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS safety_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                emergency_stop BOOLEAN DEFAULT 0,
                paused BOOLEAN DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Инициализация safety_state если пустая
        cursor.execute('SELECT COUNT(*) FROM safety_state')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO safety_state (id, emergency_stop, paused) VALUES (1, 0, 0)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def save_signal(self, trade_id, symbol, signal_type, price, indicators, ai_analysis, position_info):
        """Сохранить торговый сигнал"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO signals (
                    trade_id, symbol, signal_type, price,
                    rsi, ema20, ema50, macd, volume, avg_volume, atr,
                    filters_passed, ai_signal, ai_confidence, ai_reason,
                    amount, usdt_amount, fee, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, symbol, signal_type, price,
                indicators.get('rsi'), indicators.get('ema20'), indicators.get('ema50'),
                indicators.get('macd'), indicators.get('volume'), indicators.get('avg_volume'),
                indicators.get('atr'), indicators.get('filters_passed'),
                ai_analysis.get('signal'), ai_analysis.get('confidence'), ai_analysis.get('reason'),
                position_info.get('amount'), position_info.get('usdt_amount'),
                position_info.get('fee'), 'pending'
            ))
            conn.commit()
            logger.info(f"Signal saved: {trade_id}")
        except sqlite3.IntegrityError:
            logger.warning(f"Signal {trade_id} already exists")
        finally:
            conn.close()
    
    def update_signal_status(self, trade_id, status):
        """Обновить статус сигнала (approved/rejected)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE signals SET status = ? WHERE trade_id = ?', (status, trade_id))
        conn.commit()
        conn.close()
    
    def save_trade(self, trade_id, symbol, side, entry_price, amount, usdt_amount, fee, mode='paper', stop_loss=None, take_profit=None):
        """Сохранить выполненную сделку"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO trades (
                    trade_id, symbol, side, entry_price, amount, usdt_amount, fee,
                    mode, stop_loss, take_profit, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (trade_id, symbol, side, entry_price, amount, usdt_amount, fee,
                  mode, stop_loss, take_profit, 'open'))
            conn.commit()
            logger.info(f"Trade saved: {trade_id} ({mode} mode)")
        except sqlite3.IntegrityError:
            logger.warning(f"Trade {trade_id} already exists")
        finally:
            conn.close()
    
    def close_trade(self, trade_id, exit_price, pnl, pnl_percent):
        """Закрыть сделку"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE trades 
            SET exit_price = ?, exit_time = ?, pnl = ?, pnl_percent = ?, status = 'closed'
            WHERE trade_id = ?
        ''', (exit_price, datetime.now(), pnl, pnl_percent, trade_id))
        conn.commit()
        conn.close()
        logger.info(f"Trade closed: {trade_id}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
    
    def update_stop_loss(self, trade_id, new_stop_loss):
        """Обновить trailing stop loss для открытой сделки"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE trades 
            SET stop_loss = ?
            WHERE trade_id = ? AND status = 'open'
        ''', (new_stop_loss, trade_id))
        rows_updated = cursor.rowcount
        conn.commit()
        conn.close()
        if rows_updated > 0:
            logger.debug(f"Stop loss updated for {trade_id}: ${new_stop_loss:.2f}")
        return rows_updated > 0
    
    def get_open_trades(self):
        """Получить открытые сделки"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'open'")
        trades = cursor.fetchall()
        conn.close()
        return [dict(trade) for trade in trades] if trades else []
    
    def save_emergency_stop(self, is_active):
        """Сохранить состояние emergency_stop"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE safety_state 
            SET emergency_stop = ?, last_updated = ?
            WHERE id = 1
        ''', (1 if is_active else 0, datetime.now()))
        conn.commit()
        conn.close()
        logger.info(f"Emergency stop state saved: {is_active}")
    
    def load_emergency_stop(self):
        """Загрузить состояние emergency_stop"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT emergency_stop FROM safety_state WHERE id = 1')
        result = cursor.fetchone()
        conn.close()
        return bool(result[0]) if result else False
    
    def save_paused_state(self, is_paused):
        """Сохранить состояние паузы"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE safety_state 
            SET paused = ?, last_updated = ?
            WHERE id = 1
        ''', (1 if is_paused else 0, datetime.now()))
        conn.commit()
        conn.close()
        logger.info(f"Paused state saved: {is_paused}")
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trades WHERE status = "open" ORDER BY entry_time DESC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_all_trades(self, limit=100):
        """Получить все сделки"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_closed_trades_since(self, days=7):
        """Получить закрытые сделки за последние N дней"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM trades 
            WHERE status = 'closed' 
            AND exit_time >= datetime('now', '-' || ? || ' days')
            ORDER BY exit_time DESC
        ''', (days,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_statistics(self):
        """Получить статистику торговли"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl ELSE 0 END) as avg_loss,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as max_win,
                MIN(pnl) as max_loss
            FROM trades 
            WHERE status = 'closed'
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        if stats[0] == 0:  # No trades
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0
            }
        
        win_rate = (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
        
        return {
            'total_trades': stats[0],
            'winning_trades': stats[1] or 0,
            'losing_trades': stats[2] or 0,
            'win_rate': win_rate,
            'avg_win': stats[3] or 0,
            'avg_loss': stats[4] or 0,
            'total_pnl': stats[5] or 0,
            'avg_pnl': stats[6] or 0,
            'max_win': stats[7] or 0,
            'max_loss': stats[8] or 0
        }
    
    def get_pending_signals(self):
        """Получить ожидающие сигналы"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM signals WHERE status = "pending" ORDER BY timestamp DESC')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
