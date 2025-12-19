# NexusTrader - Cloud Deployment Guide

## 🚀 Быстрый деплой на Railway.app (рекомендую)

### Шаг 1: Регистрация
1. Зайти на https://railway.app
2. Sign up with GitHub

### Шаг 2: Деплой
1. New Project → Deploy from GitHub repo
2. Выбрать ваш репозиторий
3. Railway автоматически определит Python

### Шаг 3: Переменные окружения
В Settings → Variables добавить:
```
BINANCE_API_KEY=ваш_ключ
BINANCE_SECRET_KEY=ваш_секрет
SUPABASE_URL=ваш_url
SUPABASE_KEY=ваш_ключ
OPENAI_API_KEY=ваш_ключ
TELEGRAM_BOT_TOKEN=ваш_токен
TELEGRAM_CHAT_ID=ваш_id
PAPER_TRADING=true
AUTO_TRADE=true
MIN_CONFIDENCE=7.0
```

### Шаг 4: Запуск
Railway запустит `python trading_bot.py` автоматически

---

## 🐳 Деплой на DigitalOcean (Docker)

### Шаг 1: Создать Droplet
1. https://digitalocean.com → Create Droplet
2. Выбрать Docker image или Ubuntu
3. $5/month plan достаточно

### Шаг 2: SSH и клонирование
```bash
ssh root@your-droplet-ip
git clone https://github.com/your-username/trader.git
cd trader
```

### Шаг 3: Настройка .env
```bash
cp .env.example .env
nano .env
# Добавить все ключи
```

### Шаг 4: Запуск Docker
```bash
docker-compose up -d
```

### Шаг 5: Проверка
```bash
docker-compose logs -f
```

---

## 🆓 Бесплатный вариант - Render.com

### Шаг 1: Регистрация
https://render.com → Sign up with GitHub

### Шаг 2: Новый сервис
1. New → Background Worker
2. Connect repository
3. Runtime: Python 3
4. Build: `pip install -r requirements.txt`
5. Start: `python trading_bot.py`

### Шаг 3: Переменные
Environment → Add все ключи из .env

---

## 🍓 Raspberry Pi (для дома)

### Преимущества:
- Разовая покупка ~$50
- Работает 24/7 у вас дома
- Потребляет ~5W электричества
- Полный контроль

### Установка:
```bash
# На Raspberry Pi
sudo apt update
sudo apt install python3-pip git
git clone https://github.com/your-username/trader.git
cd trader
pip3 install -r requirements.txt
cp .env.example .env
nano .env  # добавить ключи

# Запуск как сервис
sudo nano /etc/systemd/system/trader.service
```

Содержимое trader.service:
```ini
[Unit]
Description=NexusTrader Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/trader
ExecStart=/usr/bin/python3 trading_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable trader
sudo systemctl start trader
sudo systemctl status trader
```

---

## 📊 Сравнение вариантов

| Платформа | Цена | Uptime | Сложность |
|-----------|------|--------|-----------|
| Railway | $5/мес | 99.9% | ⭐ Легко |
| Render | Free* | 99% | ⭐ Легко |
| DigitalOcean | $5/мес | 99.99% | ⭐⭐ Средне |
| Hetzner | €4/мес | 99.9% | ⭐⭐ Средне |
| Raspberry Pi | $50 разово | Зависит от вас | ⭐⭐⭐ Сложнее |

*Free tier может "засыпать" если нет трафика

---

## ⚡ Рекомендация

**Для начала**: Railway.app или Render.com
- Деплой за 5 минут
- Автоматические рестарты
- Логи в браузере
- Масштабирование

**Для продакшена**: DigitalOcean или Hetzner
- Больше контроля
- Стабильнее
- Можно добавить мониторинг
