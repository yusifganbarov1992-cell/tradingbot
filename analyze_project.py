"""
Глубокий анализ проекта - найти РЕАЛЬНЫЕ проблемы
"""
import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

def main():
    print("=" * 60)
    print("🔍 ГЛУБОКИЙ АНАЛИЗ ПРОЕКТА")
    print("=" * 60)
    print()
    
    problems = []
    suggestions = []
    
    # 1. Binance
    print("1. BINANCE CONNECTION:")
    try:
        from modules.exchanges import get_exchange
        ex = get_exchange()
        bal = ex.fetch_balance()
        free_usdt = bal.get('USDT', {}).get('free', 0)
        print(f"   ✅ Connected, USDT: ${free_usdt:.2f}")
        
        if free_usdt < 50:
            problems.append(f"Низкий баланс: ${free_usdt:.2f} - нужно минимум $50 для торговли")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        problems.append(f"Binance не работает: {e}")
    
    # 2. Portfolio
    print()
    print("2. PORTFOLIO:")
    try:
        from modules.portfolio_manager import PortfolioManager
        pm = PortfolioManager()
        portfolio = pm.get_full_portfolio()
        print(f"   Total: ${portfolio['total_usd']:.2f}")
        print(f"   Spot: ${portfolio['spot_usd']:.2f}")
        print(f"   Earn: ${portfolio['earn_usd']:.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Database
    print()
    print("3. DATABASE (Supabase):")
    try:
        from database_supabase import get_supabase_client
        sb = get_supabase_client()
        trades = sb.table('trades').select('*').execute()
        signals = sb.table('signals').select('*').execute()
        print(f"   ✅ Connected")
        print(f"   Trades: {len(trades.data)}")
        print(f"   Signals: {len(signals.data)}")
        
        # Pending signals analysis
        pending = [s for s in signals.data if s.get('status') == 'pending']
        if len(pending) > 10:
            problems.append(f"{len(pending)} сигналов 'pending' - бот не обрабатывает их!")
            suggestions.append("Проверить bot.py - почему сигналы не исполняются")
        
        # Open trades
        open_trades = [t for t in trades.data if t.get('status') == 'open']
        print(f"   Open trades: {len(open_trades)}")
        for t in open_trades:
            print(f"      - {t['symbol']} {t['side']} @ ${t['entry_price']}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        problems.append(f"Database не работает: {e}")
    
    # 4. Telegram
    print()
    print("4. TELEGRAM:")
    try:
        from modules.telegram_bot import send_telegram_sync
        result = send_telegram_sync("🔍 Тест: анализ проекта")
        if result:
            print("   ✅ Working")
        else:
            print("   ❌ Failed to send")
            problems.append("Telegram не отправляет сообщения")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Bot.py analysis
    print()
    print("5. BOT.PY ANALYSIS:")
    try:
        with open('bot.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            'Main loop': 'while True' in content,
            'Signal processing': 'process_signal' in content or 'execute_signal' in content,
            'Error handling': 'try:' in content and 'except' in content,
            'Graceful shutdown': 'KeyboardInterrupt' in content or 'signal.signal' in content,
            'Logging': 'logging' in content or 'logger' in content,
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
            if not passed:
                suggestions.append(f"bot.py: добавить {check}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 6. Safety
    print()
    print("6. SAFETY SETTINGS:")
    paper = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    auto = os.getenv('AUTO_TRADE', 'false').lower() == 'true'
    pos_size = float(os.getenv('POSITION_SIZE', '0.02'))
    min_conf = float(os.getenv('MIN_CONFIDENCE', '7.0'))
    
    print(f"   PAPER_TRADING: {'✅ ON (safe)' if paper else '⚠️ OFF - REAL MONEY!'}")
    print(f"   AUTO_TRADE: {'ON' if auto else 'OFF'}")
    print(f"   POSITION_SIZE: {pos_size*100}%")
    print(f"   MIN_CONFIDENCE: {min_conf}/10")
    
    if not paper and auto:
        problems.append("ОПАСНО: AUTO_TRADE включен с реальными деньгами!")
    
    # 7. Check if bot is running
    print()
    print("7. BOT STATUS:")
    import subprocess
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], capture_output=True, text=True)
    python_count = result.stdout.count('python.exe')
    if python_count > 1:
        print(f"   ✅ Python processes running: {python_count}")
    else:
        print("   ⚠️ Bot may not be running")
        suggestions.append("Запустить бота: python bot.py")
    
    # SUMMARY
    print()
    print("=" * 60)
    print("📋 ИТОГИ")
    print("=" * 60)
    
    if problems:
        print()
        print("❌ ПРОБЛЕМЫ (нужно исправить):")
        for i, p in enumerate(problems, 1):
            print(f"   {i}. {p}")
    else:
        print()
        print("✅ Критических проблем нет!")
    
    if suggestions:
        print()
        print("💡 РЕКОМЕНДАЦИИ (nice to have):")
        for i, s in enumerate(suggestions, 1):
            print(f"   {i}. {s}")
    
    print()
    print("=" * 60)
    
    return problems, suggestions

if __name__ == "__main__":
    problems, suggestions = main()
