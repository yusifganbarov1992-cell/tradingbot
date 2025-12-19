import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from modules.portfolio_manager import PortfolioManager

manager = PortfolioManager()
print(manager.get_portfolio_report())
