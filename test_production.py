#!/usr/bin/env python3
"""
🧪 NexusTrader AI v3.0 - Complete System Test
Tests all components to verify production readiness
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class SystemTester:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.passed = 0
        self.failed = 0
    
    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log a test result"""
        self.results.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"{GREEN}✓{RESET} {name}")
            if message:
                print(f"  {message}")
        else:
            self.failed += 1
            print(f"{RED}✗{RESET} {name}")
            if message:
                print(f"  {RED}{message}{RESET}")
    
    def print_header(self, text: str):
        """Print a section header"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{text:^60}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
    
    async def test_environment(self):
        """Test environment variables"""
        self.print_header("Testing Environment Configuration")
        
        required_vars = [
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'SUPABASE_URL',
            'SUPABASE_KEY',
            'TELEGRAM_TOKEN',
            'TELEGRAM_CHAT_ID'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if value and not value.startswith('your_'):
                self.log_test(f"Environment: {var}", True)
            else:
                self.log_test(f"Environment: {var}", False, "Not set or using template value")
    
    async def test_imports(self):
        """Test Python module imports"""
        self.print_header("Testing Module Imports")
        
        modules = [
            ('ccxt', 'Exchange API'),
            ('supabase', 'Database'),
            ('telegram', 'Telegram Bot'),
            ('streamlit', 'Dashboard'),
            ('plotly', 'Charts'),
            ('pandas', 'Data Processing'),
            ('numpy', 'Numerical Computing'),
            ('torch', 'PyTorch (AI)'),
        ]
        
        for module_name, description in modules:
            try:
                __import__(module_name)
                self.log_test(f"Import: {description}", True)
            except ImportError as e:
                self.log_test(f"Import: {description}", False, str(e))
    
    async def test_database(self):
        """Test database connection"""
        self.print_header("Testing Database Connection")
        
        try:
            from database_supabase import SupabaseClient
            
            db = SupabaseClient()
            self.log_test("Database: Connection", True)
            
            # Test query
            try:
                trades = await db.get_recent_trades(limit=1)
                self.log_test("Database: Query", True, f"Retrieved {len(trades)} trade(s)")
            except Exception as e:
                self.log_test("Database: Query", False, str(e))
                
        except Exception as e:
            self.log_test("Database: Connection", False, str(e))
    
    async def test_exchange(self):
        """Test exchange connection"""
        self.print_header("Testing Exchange Connection")
        
        try:
            import ccxt
            
            # Test public API
            exchange = ccxt.binance()
            ticker = exchange.fetch_ticker('BTC/USDT')
            self.log_test("Exchange: Public API", True, f"BTC price: ${ticker['last']:,.2f}")
            
            # Test private API
            try:
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_API_SECRET')
                
                if api_key and api_secret and not api_key.startswith('your_'):
                    exchange_auth = ccxt.binance({
                        'apiKey': api_key,
                        'secret': api_secret
                    })
                    balance = exchange_auth.fetch_balance()
                    self.log_test("Exchange: Private API", True, "Authentication successful")
                else:
                    self.log_test("Exchange: Private API", False, "API keys not configured")
            except Exception as e:
                self.log_test("Exchange: Private API", False, str(e))
                
        except Exception as e:
            self.log_test("Exchange: Public API", False, str(e))
    
    async def test_telegram(self):
        """Test Telegram connection"""
        self.print_header("Testing Telegram Connection")
        
        try:
            from telegram import Bot
            
            token = os.getenv('TELEGRAM_TOKEN')
            if token and not token.startswith('your_'):
                bot = Bot(token=token)
                me = await bot.get_me()
                self.log_test("Telegram: Bot Connection", True, f"Bot: @{me.username}")
                
                # Test send message
                try:
                    chat_id = os.getenv('TELEGRAM_CHAT_ID')
                    if chat_id:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="🧪 NexusTrader AI v3.0 - System Test\n\nAll systems operational!"
                        )
                        self.log_test("Telegram: Send Message", True)
                    else:
                        self.log_test("Telegram: Send Message", False, "Chat ID not configured")
                except Exception as e:
                    self.log_test("Telegram: Send Message", False, str(e))
            else:
                self.log_test("Telegram: Bot Connection", False, "Token not configured")
                
        except Exception as e:
            self.log_test("Telegram: Bot Connection", False, str(e))
    
    async def test_ai_modules(self):
        """Test AI modules"""
        self.print_header("Testing AI Modules")
        
        # Test LSTM
        try:
            from modules.intelligent_ai import IntelligentAI
            ai = IntelligentAI()
            self.log_test("AI: IntelligentAI Module", True)
        except Exception as e:
            self.log_test("AI: IntelligentAI Module", False, str(e))
        
        # Test Risk Manager
        try:
            from modules.risk_manager import AdvancedRiskManager
            risk = AdvancedRiskManager()
            self.log_test("AI: Risk Manager Module", True)
        except Exception as e:
            self.log_test("AI: Risk Manager Module", False, str(e))
        
        # Test Sentiment
        try:
            from modules.sentiment_analyzer import SentimentAnalyzer
            sentiment = SentimentAnalyzer()
            self.log_test("AI: Sentiment Analyzer", True)
        except Exception as e:
            self.log_test("AI: Sentiment Analyzer", False, str(e))
        
        # Test Market Regime
        try:
            from modules.market_regime import MarketRegimeManager
            regime = MarketRegimeManager()
            self.log_test("AI: Market Regime Manager", True)
        except Exception as e:
            self.log_test("AI: Market Regime Manager", False, str(e))
    
    async def test_dashboard(self):
        """Test dashboard components"""
        self.print_header("Testing Dashboard Components")
        
        try:
            from dashboard.data_provider import DashboardDataProvider, get_data_provider
            
            provider = get_data_provider()
            self.log_test("Dashboard: Data Provider", True)
            
            # Test portfolio summary
            try:
                portfolio = await provider.get_portfolio_summary()
                self.log_test("Dashboard: Portfolio Data", True, 
                            f"Balance: ${portfolio.get('balance', 0):.2f}")
            except Exception as e:
                self.log_test("Dashboard: Portfolio Data", False, str(e))
            
        except Exception as e:
            self.log_test("Dashboard: Data Provider", False, str(e))
    
    async def test_docker(self):
        """Test Docker setup"""
        self.print_header("Testing Docker Configuration")
        
        # Check if Docker files exist
        files = [
            'Dockerfile.bot',
            'Dockerfile.dashboard',
            'docker-compose.yml',
            '.env.template'
        ]
        
        for file in files:
            if os.path.exists(file):
                self.log_test(f"Docker: {file}", True)
            else:
                self.log_test(f"Docker: {file}", False, "File not found")
    
    async def run_all_tests(self):
        """Run all tests"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{'NexusTrader AI v3.0 - System Test':^60}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"\n{YELLOW}Starting comprehensive system test...{RESET}")
        print(f"{YELLOW}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}\n")
        
        # Run all test suites
        await self.test_environment()
        await self.test_imports()
        await self.test_database()
        await self.test_exchange()
        await self.test_telegram()
        await self.test_ai_modules()
        await self.test_dashboard()
        await self.test_docker()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}{'Test Summary':^60}{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"Success Rate: {percentage:.1f}%\n")
        
        if self.failed == 0:
            print(f"{GREEN}{'='*60}{RESET}")
            print(f"{GREEN}{'✓ ALL TESTS PASSED! SYSTEM READY FOR DEPLOYMENT':^60}{RESET}")
            print(f"{GREEN}{'='*60}{RESET}\n")
            return 0
        else:
            print(f"{YELLOW}{'='*60}{RESET}")
            print(f"{YELLOW}{'⚠ SOME TESTS FAILED - REVIEW ERRORS ABOVE':^60}{RESET}")
            print(f"{YELLOW}{'='*60}{RESET}\n")
            
            print("Failed Tests:")
            for name, passed, message in self.results:
                if not passed:
                    print(f"  {RED}✗{RESET} {name}")
                    if message:
                        print(f"    {message}")
            
            print()
            return 1

async def main():
    """Main test function"""
    tester = SystemTester()
    exit_code = await tester.run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run tests
    asyncio.run(main())
