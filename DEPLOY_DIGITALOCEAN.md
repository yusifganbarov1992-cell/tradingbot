# 🚀 Деплой на DigitalOcean

## Шаг 1: Создать Droplet

1. Зайди на [DigitalOcean](https://cloud.digitalocean.com/)
2. **Create** → **Droplets**
3. Выбери:
   - **Image**: Ubuntu 24.04 LTS
   - **Plan**: Basic ($6/month - 1GB RAM достаточно)
   - **Region**: Ближайший к тебе (Frankfurt/Amsterdam для Европы)
   - **Authentication**: SSH Key (создай если нет) или Password
4. Нажми **Create Droplet**
5. Скопируй IP адрес (например: `165.227.XXX.XXX`)

---

## Шаг 2: Подключиться к серверу

### Windows (PowerShell):
```powershell
ssh root@165.227.XXX.XXX
```

Введи пароль (если используешь password auth)

---

## Шаг 3: Установить зависимости на сервере

```bash
# Обновить систему
apt update && apt upgrade -y

# Установить Python 3.11+
apt install python3 python3-pip python3-venv git -y

# Проверить версию
python3 --version
```

---

## Шаг 4: Загрузить бота на сервер

### Вариант A: Через Git (рекомендую)

**На своём ПК:**
```bash
# Создать репозиторий на GitHub
# Загрузить код (инструкция ниже в разделе GitHub)
```

**На сервере:**
```bash
cd /opt
git clone https://github.com/ТВОЙ_USERNAME/trader.git
cd trader
```

### Вариант B: Через SCP (если нет GitHub)

**На своём ПК (PowerShell):**
```powershell
scp -r C:\Users\yusif\OneDrive\Desktop\trader root@165.227.XXX.XXX:/opt/
```

---

## Шаг 5: Настроить окружение на сервере

```bash
cd /opt/trader

# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Создать .env файл с ключами
nano .env
```

**Скопируй содержимое локального файла .env в файл .env на сервере:**
```bash
nano .env
# Вставь все свои реальные API ключи
# Ctrl+X, затем Y, затем Enter для сохранения
```

Сохрани: `Ctrl+X` → `Y` → `Enter`

---

## Шаг 6: Запустить бота в фоне (Screen)

```bash
# Установить screen
apt install screen -y

# Создать сессию
screen -S trading_bot

# Запустить бота
source .venv/bin/activate
python trading_bot.py

# Увидишь:
# 🛡️ SafetyManager initialized...
# ✅ Bot is running 24/7...
```

**Отключиться от сессии (бот продолжит работать):**
Нажми: `Ctrl+A` потом `D`

**Вернуться к боту:**
```bash
screen -r trading_bot
```

**Посмотреть все сессии:**
```bash
screen -ls
```

---

## Шаг 7: Автозапуск при перезагрузке (systemd)

Создай сервис:
```bash
nano /etc/systemd/system/trading_bot.service
```

Вставь:
```ini
[Unit]
Description=AI Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/trader
Environment="PATH=/opt/trader/.venv/bin"
ExecStart=/opt/trader/.venv/bin/python trading_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Сохрани и активируй:
```bash
systemctl daemon-reload
systemctl enable trading_bot
systemctl start trading_bot
systemctl status trading_bot
```

---

## 📊 Мониторинг

### Проверить статус:
```bash
systemctl status trading_bot
```

### Посмотреть логи:
```bash
journalctl -u trading_bot -f
```

### Остановить бота:
```bash
systemctl stop trading_bot
```

### Перезапустить:
```bash
systemctl restart trading_bot
```

---

## 🔒 Безопасность

```bash
# Отключить root login по SSH
nano /etc/ssh/sshd_config
# Найти: PermitRootLogin yes
# Изменить на: PermitRootLogin no

# Настроить firewall
ufw allow 22/tcp
ufw enable
```

---

## 🎯 Итог

✅ Бот работает 24/7 на сервере
✅ Автоматически перезапускается при сбоях
✅ Логи доступны через journalctl
✅ Можно закрыть ноутбук - всё работает!

**Стоимость:** $6/месяц (оправдано если бот торгует)
