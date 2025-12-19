#!/bin/bash

# 🚀 NexusTrader AI v3.0 - One-Click Deployment Script
# This script sets up and deploys the complete trading system

set -e  # Exit on error

echo "========================================"
echo "🚀 NexusTrader AI v3.0 Deployment"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check prerequisites
echo -e "${BLUE}Step 1: Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found! Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found! Please install Docker Compose first.${NC}"
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker found${NC}"
echo -e "${GREEN}✓ Docker Compose found${NC}"
echo ""

# Step 2: Check .env file
echo -e "${BLUE}Step 2: Checking environment configuration...${NC}"

if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠  .env file not found. Creating from template...${NC}"
    if [ -f .env.template ]; then
        cp .env.template .env
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠  Please edit .env file with your API keys before continuing!${NC}"
        echo ""
        read -p "Press Enter after editing .env file..."
    else
        echo -e "${RED}✗ .env.template not found!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Verify required env vars
REQUIRED_VARS="BINANCE_API_KEY BINANCE_API_SECRET SUPABASE_URL SUPABASE_KEY TELEGRAM_TOKEN TELEGRAM_CHAT_ID"
MISSING_VARS=""

for VAR in $REQUIRED_VARS; do
    if ! grep -q "^${VAR}=" .env || grep -q "^${VAR}=your_" .env || grep -q "^${VAR}=$" .env; then
        MISSING_VARS="${MISSING_VARS} ${VAR}"
    fi
done

if [ -n "$MISSING_VARS" ]; then
    echo -e "${RED}✗ Missing or incomplete environment variables:${MISSING_VARS}${NC}"
    echo -e "${YELLOW}Please edit .env file and set these variables.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All required environment variables set${NC}"
echo ""

# Step 3: Create directories
echo -e "${BLUE}Step 3: Creating directories...${NC}"

mkdir -p logs
mkdir -p models
mkdir -p dashboard

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 4: Test connections (optional)
echo -e "${BLUE}Step 4: Testing API connections...${NC}"
echo -e "${YELLOW}Do you want to test API connections before deployment? (y/n)${NC}"
read -p "> " TEST_APIS

if [ "$TEST_APIS" = "y" ] || [ "$TEST_APIS" = "Y" ]; then
    echo "Testing Binance connection..."
    if python test_binance.py 2>/dev/null; then
        echo -e "${GREEN}✓ Binance connection OK${NC}"
    else
        echo -e "${YELLOW}⚠  Binance test failed. Continue anyway? (y/n)${NC}"
        read -p "> " CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            exit 1
        fi
    fi

    echo "Testing Supabase connection..."
    if python test_simple_supabase.py 2>/dev/null; then
        echo -e "${GREEN}✓ Supabase connection OK${NC}"
    else
        echo -e "${YELLOW}⚠  Supabase test failed. Continue anyway? (y/n)${NC}"
        read -p "> " CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            exit 1
        fi
    fi

    echo "Testing Telegram connection..."
    if python check_telegram.py 2>/dev/null; then
        echo -e "${GREEN}✓ Telegram connection OK${NC}"
    else
        echo -e "${YELLOW}⚠  Telegram test failed. Continue anyway? (y/n)${NC}"
        read -p "> " CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            exit 1
        fi
    fi
fi
echo ""

# Step 5: Build Docker images
echo -e "${BLUE}Step 5: Building Docker images...${NC}"
echo "This may take several minutes..."

if docker-compose build; then
    echo -e "${GREEN}✓ Docker images built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed!${NC}"
    exit 1
fi
echo ""

# Step 6: Start services
echo -e "${BLUE}Step 6: Starting services...${NC}"

if docker-compose up -d; then
    echo -e "${GREEN}✓ Services started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start services!${NC}"
    exit 1
fi
echo ""

# Step 7: Wait for services to be healthy
echo -e "${BLUE}Step 7: Waiting for services to be healthy...${NC}"
echo "This may take 30-60 seconds..."

sleep 10  # Give containers time to start

MAX_WAIT=60
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    HEALTHY=$(docker-compose ps | grep -c "healthy" || true)
    TOTAL=$(docker-compose ps | grep -c "Up" || true)
    
    if [ $HEALTHY -eq $TOTAL ] && [ $TOTAL -gt 0 ]; then
        echo -e "${GREEN}✓ All services are healthy${NC}"
        break
    fi
    
    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${YELLOW}⚠  Services may not be fully healthy yet. Check logs with: docker-compose logs${NC}"
fi
echo ""

# Step 8: Display status
echo -e "${BLUE}Step 8: Deployment status...${NC}"
docker-compose ps
echo ""

# Step 9: Show access information
echo -e "${GREEN}========================================"
echo "🎉 Deployment Complete!"
echo "========================================${NC}"
echo ""
echo -e "${BLUE}Access Points:${NC}"
echo "• Dashboard: http://localhost:8501"
echo "• Bot logs: docker-compose logs -f trading-bot"
echo "• Dashboard logs: docker-compose logs -f dashboard"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "• View logs: docker-compose logs -f"
echo "• Stop services: docker-compose down"
echo "• Restart: docker-compose restart"
echo "• View status: docker-compose ps"
echo ""
echo -e "${BLUE}Telegram Commands:${NC}"
echo "• /start - Start bot"
echo "• /status - Check status"
echo "• /balance - Check balance"
echo "• /help - All commands"
echo ""
echo -e "${YELLOW}⚠  IMPORTANT REMINDERS:${NC}"
echo "1. Bot is in PAPER TRADING mode by default"
echo "2. Set AUTO_TRADE=true in .env to enable auto-trading"
echo "3. Monitor the dashboard and Telegram closely"
echo "4. Start with small amounts when going live"
echo "5. Review trades regularly"
echo ""
echo -e "${GREEN}Good luck and happy trading! 🚀${NC}"
echo ""

# Step 10: Open dashboard (optional)
echo -e "${BLUE}Open dashboard in browser? (y/n)${NC}"
read -p "> " OPEN_BROWSER

if [ "$OPEN_BROWSER" = "y" ] || [ "$OPEN_BROWSER" = "Y" ]; then
    # Detect OS and open browser
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:8501 2>/dev/null || true
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:8501
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        start http://localhost:8501
    fi
fi

echo ""
echo -e "${GREEN}Deployment script completed successfully!${NC}"
