# 🤝 Contributing to NexusTrader AI

Thank you for your interest in contributing to NexusTrader AI! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Pull Request Process](#pull-request-process)
6. [Code Style](#code-style)
7. [Testing](#testing)
8. [Documentation](#documentation)

---

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the best outcome for the project
- Follow security best practices (never commit secrets)

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment (venv or conda)

### Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/nexustrader-ai.git
cd nexustrader-ai
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"  # Install dev dependencies
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (for testing only)
```

### 4. Verify Setup

```bash
python -c "from trading_bot import TradingAgent; print('Setup OK')"
pytest --collect-only  # Check tests can be found
```

---

## Making Changes

### Branch Naming

```
feature/add-new-indicator
bugfix/fix-trade-execution
docs/update-readme
refactor/improve-safety-manager
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add RSI indicator to analysis
fix: correct position size calculation
docs: update API documentation
test: add unit tests for SafetyManager
refactor: simplify trade executor logic
chore: update dependencies
```

### Example Workflow

```bash
# Create feature branch
git checkout -b feature/add-new-indicator

# Make changes
# ... edit files ...

# Run tests
pytest

# Run linting
black .
flake8 .

# Commit
git add .
git commit -m "feat: add RSI indicator to analysis"

# Push
git push origin feature/add-new-indicator
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`black .`)
- [ ] Linting passes (`flake8 .`)
- [ ] Documentation updated (if applicable)
- [ ] No hardcoded secrets
- [ ] Branch is up to date with `main`

### 2. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
```

### 3. Review Process

1. Submit PR to `main` branch
2. Automated CI checks run
3. Code review by maintainers
4. Address feedback
5. Merge when approved

---

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these tools:

```bash
# Format code
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Style Rules

- Line length: 100 characters
- Use type hints for function signatures
- Write docstrings for public functions
- Use meaningful variable names

### Example

```python
from typing import Dict, Optional

def analyze_market(symbol: str, timeframe: str = "1h") -> Optional[Dict]:
    """
    Analyze market data for a given symbol.
    
    Args:
        symbol: Trading pair (e.g., "BTC/USDT")
        timeframe: Candlestick timeframe
        
    Returns:
        Analysis dictionary or None if failed
    """
    try:
        # Implementation
        return {"signal": "BUY", "confidence": 7.5}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest test_trading_bot.py

# Specific test
pytest test_trading_bot.py::test_analyze_market
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_analyze_market_success():
    """Test successful market analysis."""
    agent = TradingAgent()
    
    with patch.object(agent, 'get_market_data') as mock_data:
        mock_data.return_value = {"price": 88000}
        result = agent.analyze_market_symbol("BTC/USDT")
        
    assert result is not None
    assert "signal" in result

def test_analyze_market_invalid_symbol():
    """Test handling of invalid symbol."""
    agent = TradingAgent()
    result = agent.analyze_market_symbol("INVALID")
    assert result is None
```

### Test Categories

- `test_*.py` - Unit tests
- `test_integration_*.py` - Integration tests
- `test_e2e_*.py` - End-to-end tests

---

## Documentation

### Updating Docs

- `README.md` - Project overview
- `API.md` - API reference
- `ARCHITECTURE.md` - System design
- Code docstrings - Inline documentation

### Docstring Format

```python
def function_name(param1: str, param2: int = 10) -> Dict:
    """
    Short description of function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        
    Example:
        >>> result = function_name("test", 5)
        >>> print(result)
        {'status': 'ok'}
    """
    pass
```

---

## Need Help?

- Open an [Issue](https://github.com/nexustrader/nexustrader-ai/issues)
- Read the [Documentation](./README.md)
- Check existing [Pull Requests](https://github.com/nexustrader/nexustrader-ai/pulls)

---

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- GitHub Contributors page

Thank you for contributing! 🎉
