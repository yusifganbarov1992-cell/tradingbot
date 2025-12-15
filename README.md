# NexusTrader AI

## Architecture
This project consists of two parts:
1. **Frontend (React):** The visual interface (runs in browser).
2. **Backend (Python):** The trading bot (runs on your server/computer).

---

## 🚀 Server Installation Guide (The "Easy" Way)

### 1. Prepare files
Upload `trading_bot.py` and `requirements.txt` to your server folder.

### 2. Install Dependencies
Run this command in your terminal to install everything automatically:
```bash
pip install -r requirements.txt
```

### 3. Configure Credentials
You have two options:
1.  **Option A (Secure):** Create a `.env` file in the same folder:
    ```
    BINANCE_API_KEY=your_key_here
    BINANCE_SECRET_KEY=your_secret_here
    TELEGRAM_BOT_TOKEN=your_token_here
    ```
2.  **Option B (Quick):** Edit `trading_bot.py` lines 15-17 directly.

### 4. Run the Bot
```bash
python trading_bot.py
```

### 5. How to use
1.  Open Telegram.
2.  Send `/start` to wake the bot.
3.  Send `/train` to initialize the AI (if libraries are installed).
4.  Send `/analyze` to get a real-time signal.

---

## 🛠 Features & Improvements

-   **Hybrid Mode:** If your server cannot run TensorFlow, the bot automatically switches to "Indicator Mode" (RSI/EMA only) so it never crashes.
-   **Async Core:** Uses the latest Telegram libraries for maximum speed.
-   **Safety:** Requires manual confirmation (Buttons) for every trade.

