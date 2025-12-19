"""
Test autostart configuration
"""
import os
import subprocess

print("=" * 60)
print("    AUTOSTART CONFIGURATION CHECK")
print("=" * 60)
print()

# Check startup shortcut
startup_path = os.path.join(
    os.environ['APPDATA'],
    'Microsoft', 'Windows', 'Start Menu', 'Programs', 'Startup',
    'NexusTrader.lnk'
)

if os.path.exists(startup_path):
    print(f"✅ Startup shortcut exists:")
    print(f"   {startup_path}")
else:
    print(f"❌ Startup shortcut NOT FOUND!")
    print(f"   Expected: {startup_path}")

print()

# Check START_BOT.bat
if os.path.exists('START_BOT.bat'):
    print("✅ START_BOT.bat exists")
else:
    print("❌ START_BOT.bat NOT FOUND!")

# Check start_hidden.vbs
if os.path.exists('start_hidden.vbs'):
    print("✅ start_hidden.vbs exists (hidden mode)")
else:
    print("⚠️ start_hidden.vbs not found (will show console)")

# Check bot_watchdog.py
if os.path.exists('bot_watchdog.py'):
    print("✅ bot_watchdog.py exists")
else:
    print("❌ bot_watchdog.py NOT FOUND!")

# Check trading_bot.py
if os.path.exists('trading_bot.py'):
    print("✅ trading_bot.py exists")
else:
    print("❌ trading_bot.py NOT FOUND!")

print()

# Check if bot is running
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                       capture_output=True, text=True)
if 'python.exe' in result.stdout:
    lines = [l for l in result.stdout.split('\n') if 'python.exe' in l.lower()]
    print(f"✅ Python processes running: {len(lines)}")
else:
    print("⚠️ No Python processes running")

print()
print("=" * 60)
print("    HOW AUTOSTART WORKS:")
print("=" * 60)
print("""
1. Windows starts → Runs NexusTrader.lnk from Startup folder
2. NexusTrader.lnk → Runs start_hidden.vbs (no console window)
3. start_hidden.vbs → Runs START_BOT.bat
4. START_BOT.bat → Activates venv, runs bot_watchdog.py
5. bot_watchdog.py → Runs trading_bot.py, restarts on crash
""")

print("=" * 60)
print("    TO TEST AUTOSTART:")
print("=" * 60)
print("""
Option 1: Restart computer
Option 2: Run manually:
   wscript.exe start_hidden.vbs
   
To stop bot completely:
   taskkill /F /IM python.exe
""")
