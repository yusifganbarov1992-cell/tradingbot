"""
Watchdog для 24/7 работы бота
Перезапускает бот при любой остановке
"""
import subprocess
import time
import sys
from datetime import datetime

def print_banner():
    print("=" * 70)
    print("  NEXUSTRADER WATCHDOG - 24/7 OPERATION")
    print("  Bot will restart automatically on any stop/crash")
    print("  Press Ctrl+C to stop completely")
    print("=" * 70)
    print()

def run_bot():
    """Запустить бот и вернуть exit code"""
    process = subprocess.Popen(
        [sys.executable, 'trading_bot.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    
    # Выводим логи в реальном времени
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
    except UnicodeDecodeError:
        pass
    
    process.wait()
    return process.returncode

def main():
    print_banner()
    
    restart_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            restart_count += 1
            uptime = datetime.now() - start_time
            
            print(f"\n{'='*70}")
            print(f"  BOT START #{restart_count}")
            print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Total uptime: {uptime}")
            print(f"{'='*70}\n")
            
            exit_code = run_bot()
            
            print(f"\n{'='*70}")
            print(f"  BOT STOPPED")
            print(f"  Exit code: {exit_code}")
            print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if exit_code != 0:
                print(f"  ⚠️  Bot crashed!")
            else:
                print(f"  ℹ️  Bot stopped normally")
            
            print(f"  Restarting in 3 seconds...")
            print(f"{'='*70}\n")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  SHUTDOWN REQUESTED")
        print(f"  Total restarts: {restart_count}")
        print(f"  Total uptime: {datetime.now() - start_time}")
        print("  Goodbye!")
        print("="*70)
        sys.exit(0)

if __name__ == '__main__':
    main()
