"""
Launcher script for NexusTrader AI
Starts all components: Trading Bot, API Server, and Frontend
"""
import subprocess
import sys
import os
import time
from pathlib import Path

def start_component(name, command, cwd=None):
    """Start a component in a new process"""
    print(f"🚀 Starting {name}...")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)  # Give it time to start
        if process.poll() is None:
            print(f"✅ {name} started successfully (PID: {process.pid})")
            return process
        else:
            print(f"❌ {name} failed to start")
            return None
    except Exception as e:
        print(f"❌ Error starting {name}: {e}")
        return None

def main():
    print("=" * 60)
    print("🤖 NexusTrader AI - Full Stack Launcher")
    print("=" * 60)
    print()
    
    base_dir = Path(__file__).parent
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    
    # Check environment
    if not venv_python.exists():
        print("❌ Virtual environment not found. Run setup first.")
        return
    
    env_file = base_dir / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found. Some features may not work.")
    
    print("📋 Starting components...\n")
    
    processes = []
    
    # 1. Start API Server
    api_cmd = f'"{venv_python}" api_server.py'
    api_process = start_component("API Server", api_cmd, cwd=base_dir)
    if api_process:
        processes.append(("API Server", api_process))
    
    time.sleep(3)
    
    # 2. Start Trading Bot
    bot_cmd = f'"{venv_python}" trading_bot.py'
    bot_process = start_component("Trading Bot", bot_cmd, cwd=base_dir)
    if bot_process:
        processes.append(("Trading Bot", bot_process))
    
    time.sleep(3)
    
    # 3. Start Frontend
    frontend_cmd = "npm run dev"
    frontend_process = start_component("Frontend", frontend_cmd, cwd=base_dir)
    if frontend_process:
        processes.append(("Frontend", frontend_process))
    
    print()
    print("=" * 60)
    print("✅ All components started!")
    print("=" * 60)
    print()
    print("📊 Access points:")
    print("   • Web Dashboard: http://localhost:3000")
    print("   • API Server: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Telegram Bot: Check your Telegram")
    print()
    print("Press Ctrl+C to stop all components...")
    print()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {name} stopped unexpectedly")
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all components...")
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
                print(f"   • {name} stopped")
            except:
                proc.kill()
                print(f"   • {name} force killed")
        print("\n✅ All components stopped. Goodbye!")

if __name__ == "__main__":
    main()
