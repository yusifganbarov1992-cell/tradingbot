# 🚀 Production Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Testing](#local-testing)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required:
- Docker & Docker Compose
- Python 3.11+
- Git
- Binance API keys
- Supabase account
- Telegram Bot token

### Optional:
- Domain name
- SSL certificate
- VPN/VPS server
- Monitoring tools (Grafana, Prometheus)

---

## Local Testing

### 1. Clone Repository:
```bash
git clone <your-repo-url>
cd trader
```

### 2. Setup Environment:
```bash
# Copy template
cp .env.template .env

# Edit with your keys
nano .env
```

### 3. Install Dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install
pip install -r requirements.txt
pip install -r requirements_new.txt
```

### 4. Test Components:

#### Test Database Connection:
```bash
python test_simple_supabase.py
```

#### Test Exchange Connection:
```bash
python test_binance.py
```

#### Test Bot (Paper Trading):
```bash
# Make sure AUTO_TRADE=false in .env
python trading_bot.py
```

#### Test Dashboard:
```bash
streamlit run dashboard/app.py
```

### 5. Verify Everything:
- ✅ Database connection successful
- ✅ Binance API working
- ✅ Telegram notifications received
- ✅ Dashboard loads without errors
- ✅ All modules import successfully

---

## Docker Deployment

### 1. Build Images:
```bash
# Build all services
docker-compose build

# Or build individually
docker-compose build trading-bot
docker-compose build dashboard
```

### 2. Start Services:
```bash
# Start all in background
docker-compose up -d

# Start with logs
docker-compose up

# Start specific service
docker-compose up -d dashboard
```

### 3. View Logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f trading-bot
docker-compose logs -f dashboard

# Last 100 lines
docker-compose logs --tail=100 trading-bot
```

### 4. Check Status:
```bash
# List running containers
docker-compose ps

# Check health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Resource usage
docker stats
```

### 5. Stop Services:
```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart service
docker-compose restart trading-bot
```

### 6. Update Deployment:
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d

# Or rolling update
docker-compose up -d --build
```

---

## Cloud Deployment

### Option 1: DigitalOcean Droplet

#### 1.1 Create Droplet:
```bash
# Recommended specs:
- OS: Ubuntu 22.04 LTS
- Plan: Basic ($12/month or higher)
- CPU: 2 vCPUs
- RAM: 2 GB
- Storage: 50 GB SSD
- Region: Closest to you
```

#### 1.2 Initial Server Setup:
```bash
# SSH into server
ssh root@your_server_ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Create non-root user
adduser trader
usermod -aG sudo trader
usermod -aG docker trader

# Switch to user
su - trader
```

#### 1.3 Deploy Application:
```bash
# Clone repository
git clone <your-repo-url>
cd trader

# Setup environment
nano .env
# (paste your configuration)

# Start services
docker-compose up -d

# Verify
docker-compose ps
```

#### 1.4 Setup Firewall:
```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow Dashboard (if public)
sudo ufw allow 8501/tcp

# Check status
sudo ufw status
```

#### 1.5 Setup Domain (Optional):
```bash
# Install Nginx
sudo apt install nginx -y

# Create config
sudo nano /etc/nginx/sites-available/dashboard

# Add:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Setup SSL (Let's Encrypt)
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

### Option 2: AWS EC2

#### 2.1 Launch Instance:
```bash
# Recommended:
- AMI: Ubuntu 22.04 LTS
- Instance Type: t3.small or t3.medium
- Storage: 30 GB GP3
- Security Group: Allow 22, 8501, 80, 443
```

#### 2.2 Connect & Setup:
```bash
# SSH
ssh -i your-key.pem ubuntu@ec2-ip-address

# Same setup as DigitalOcean above
```

---

### Option 3: Google Cloud Platform

#### 3.1 Create VM Instance:
```bash
# gcloud CLI
gcloud compute instances create trading-bot \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=30GB

# SSH
gcloud compute ssh trading-bot

# Same setup as above
```

---

### Option 4: Docker Hub + Cloud Run

#### 4.1 Push Images:
```bash
# Login to Docker Hub
docker login

# Tag images
docker tag trader_trading-bot:latest yourusername/trading-bot:latest
docker tag trader_dashboard:latest yourusername/dashboard:latest

# Push
docker push yourusername/trading-bot:latest
docker push yourusername/dashboard:latest
```

#### 4.2 Deploy to Cloud Run (Dashboard only):
```bash
gcloud run deploy dashboard \
    --image yourusername/dashboard:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

---

## Monitoring & Maintenance

### 1. Health Monitoring:

#### Docker Health Checks:
```bash
# Check container health
docker ps --filter "health=healthy"
docker ps --filter "health=unhealthy"

# Inspect health
docker inspect --format='{{.State.Health.Status}}' crypto-trading-bot
```

#### Application Logs:
```bash
# Bot logs
docker-compose logs -f trading-bot | grep ERROR

# Dashboard logs
docker-compose logs -f dashboard | grep ERROR

# Export logs
docker-compose logs > logs_$(date +%Y%m%d).txt
```

### 2. Performance Monitoring:

#### System Resources:
```bash
# CPU, Memory, Network
docker stats

# Disk usage
df -h
docker system df

# Clean up
docker system prune -a
```

#### Application Metrics:
```bash
# Check trade count
python check_database.py

# Check balance
python check_balance.py

# Database stats
# Login to Supabase Dashboard
```

### 3. Backup Strategy:

#### Database Backups:
```bash
# Supabase has automatic backups
# Additional manual backup:
python -c "
import asyncio
from database_supabase import SupabaseClient

async def backup():
    db = SupabaseClient()
    trades = await db.get_all_trades()
    import json
    with open('backup_trades.json', 'w') as f:
        json.dump(trades, f)

asyncio.run(backup())
"
```

#### Code Backups:
```bash
# Git backup
git add .
git commit -m "Backup $(date +%Y%m%d)"
git push

# Config backup
cp .env .env.backup
```

### 4. Automated Monitoring:

#### Setup Cron Jobs:
```bash
# Edit crontab
crontab -e

# Add monitoring jobs:
# Check every 5 minutes
*/5 * * * * cd /home/trader/trader && docker-compose ps | grep -q "Up" || docker-compose up -d

# Daily backup
0 2 * * * cd /home/trader/trader && python backup_script.py

# Weekly cleanup
0 3 * * 0 cd /home/trader/trader && docker system prune -f
```

### 5. Telegram Alerts:
```python
# Already integrated in bot
# Receives:
- Trade notifications
- Error alerts
- System status
- Daily reports
```

---

## Troubleshooting

### Common Issues:

#### 1. Container Won't Start:
```bash
# Check logs
docker-compose logs trading-bot

# Common fixes:
# - Check .env file exists
# - Verify API keys
# - Check port conflicts
docker ps -a | grep 8501

# Restart
docker-compose restart trading-bot
```

#### 2. Database Connection Failed:
```bash
# Test connection
python test_simple_supabase.py

# Fixes:
# - Verify SUPABASE_URL and SUPABASE_KEY
# - Check Supabase dashboard for outages
# - Verify RLS policies
```

#### 3. Binance API Errors:
```bash
# Test API
python test_binance.py

# Common issues:
# - Invalid API key
# - IP not whitelisted
# - Rate limit exceeded
# - Insufficient permissions
```

#### 4. Dashboard Not Loading:
```bash
# Check if running
docker-compose ps dashboard

# Check logs
docker-compose logs dashboard

# Restart
docker-compose restart dashboard

# Access directly
http://localhost:8501
```

#### 5. Out of Memory:
```bash
# Check usage
docker stats

# Increase limits in docker-compose.yml:
services:
  trading-bot:
    mem_limit: 1g
    memswap_limit: 1g
```

#### 6. High CPU Usage:
```bash
# Check processes
docker exec -it crypto-trading-bot top

# Reduce scan frequency in .env:
SCAN_INTERVAL=600  # 10 minutes instead of 5
```

---

## Security Best Practices

### 1. API Keys:
- ✅ Never commit .env to Git
- ✅ Use read-only API keys when possible
- ✅ Enable IP whitelist on Binance
- ✅ Rotate keys periodically

### 2. Server Security:
```bash
# Disable root login
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no

# Use SSH keys only
# Set: PasswordAuthentication no

# Restart SSH
sudo systemctl restart sshd

# Install fail2ban
sudo apt install fail2ban -y
```

### 3. Dashboard Access:
```bash
# Option 1: VPN only
# Setup WireGuard or OpenVPN

# Option 2: IP whitelist in Nginx
location / {
    allow your_ip;
    deny all;
    proxy_pass http://localhost:8501;
}

# Option 3: Basic auth
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin
# Add to Nginx config:
auth_basic "Dashboard";
auth_basic_user_file /etc/nginx/.htpasswd;
```

### 4. Firewall:
```bash
# Minimal ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Block all other ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
```

---

## Production Checklist

### Pre-Deployment:
- [ ] All tests passing
- [ ] .env configured correctly
- [ ] API keys validated
- [ ] Database accessible
- [ ] Telegram working
- [ ] Paper trading tested

### Deployment:
- [ ] Docker containers running
- [ ] Health checks passing
- [ ] Logs show no errors
- [ ] Dashboard accessible
- [ ] First trade executed (small amount)

### Post-Deployment:
- [ ] Monitoring setup
- [ ] Backups configured
- [ ] Alerts working
- [ ] Documentation updated
- [ ] Team trained

### Security:
- [ ] Firewall enabled
- [ ] SSH keys only
- [ ] Dashboard protected
- [ ] SSL certificate installed
- [ ] API keys secured

---

## Performance Tuning

### 1. Optimize Scanning:
```env
# In .env
SCAN_INTERVAL=300        # 5 minutes
MAX_PAIRS_SCAN=30        # Reduce pairs
WORKER_THREADS=2         # Less threads
```

### 2. Database Optimization:
```sql
-- Add indexes
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
```

### 3. Docker Resources:
```yaml
# In docker-compose.yml
services:
  trading-bot:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

---

## Maintenance Schedule

### Daily:
- Check logs for errors
- Verify trades executing
- Monitor resource usage
- Review Telegram alerts

### Weekly:
- Analyze performance
- Review strategy results
- Update documentation
- Check for updates

### Monthly:
- Full system backup
- Security audit
- Performance review
- Strategy optimization
- Rotate API keys

---

## Support & Updates

### Getting Help:
1. Check logs first
2. Review documentation
3. Test components individually
4. Check GitHub issues

### Staying Updated:
```bash
# Pull latest changes
git fetch
git pull

# Rebuild containers
docker-compose build
docker-compose up -d
```

---

## Conclusion

Your trading bot is now production-ready! 🎉

**Remember:**
- Start with small amounts
- Monitor closely
- Test thoroughly
- Keep learning
- Stay safe

**Good luck! 📈🚀**

---

*Last Updated: December 16, 2024*
*Version: 1.0 Production*
