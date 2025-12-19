"""
Тест интеграции с Telegram
Проверяет отправку сообщений и кнопок
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from datetime import datetime

async def test_telegram():
    """Полный тест Telegram интеграции"""
    print("="*60)
    print("  TELEGRAM INTEGRATION TEST")
    print("="*60)
    
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('OPERATOR_CHAT_ID')
    
    if not token or not chat_id:
        print("\n❌ FAILED: Missing TELEGRAM_BOT_TOKEN or OPERATOR_CHAT_ID in .env")
        return False
    
    print(f"\nBot token: {token[:30]}...")
    print(f"Chat ID: {chat_id}")
    
    bot = Bot(token)
    
    # Test 1: Bot info
    print("\n1. Getting bot info...")
    try:
        me = await bot.get_me()
        print(f"   ✅ Connected to @{me.username}")
        print(f"      - ID: {me.id}")
        print(f"      - First name: {me.first_name}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: Simple message
    print("\n2. Sending simple message...")
    try:
        msg = await bot.send_message(
            chat_id=chat_id,
            text=f"🧪 TEST MESSAGE\nTime: {datetime.now().strftime('%H:%M:%S')}\n\nTelegram API working!"
        )
        print(f"   ✅ Message sent (ID: {msg.message_id})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Message with inline buttons
    print("\n3. Sending message with inline buttons...")
    try:
        keyboard = [[
            InlineKeyboardButton("✅ Approve", callback_data="test_approve"),
            InlineKeyboardButton("❌ Reject", callback_data="test_reject")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg2 = await bot.send_message(
            chat_id=chat_id,
            text="🧪 BUTTON TEST\n\nClick a button to test callback handling:",
            reply_markup=reply_markup
        )
        print(f"   ✅ Buttons sent (ID: {msg2.message_id})")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 4: Simulated trading signal
    print("\n4. Sending simulated trading signal...")
    try:
        trade_id = f"TEST_{int(datetime.now().timestamp())}"
        signal_message = f"""
🚨 ТЕСТОВЫЙ СИГНАЛ #{trade_id}

📊 Symbol: ETH/USDT
📈 Signal: BUY
💰 Price: $3,500.00
📦 Amount: 0.0143 (~$50.00)

🤖 AI Analysis:
Confidence: 8/10
Reason: Test signal for integration check

⚠️ ЭТО ТЕСТ! НЕ ПОДТВЕРЖДАЙТЕ!
"""
        
        keyboard = [[
            InlineKeyboardButton("✅ Подтвердить", callback_data=f"approve_{trade_id}"),
            InlineKeyboardButton("❌ Отклонить", callback_data=f"reject_{trade_id}")
        ]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        msg3 = await bot.send_message(
            chat_id=chat_id,
            text=signal_message,
            reply_markup=reply_markup
        )
        print(f"   ✅ Signal sent (ID: {msg3.message_id})")
        print(f"      Trade ID: {trade_id}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 5: Get recent updates
    print("\n5. Checking recent updates...")
    try:
        updates = await bot.get_updates(limit=5)
        print(f"   ✅ Received {len(updates)} recent updates")
        
        if updates:
            latest = updates[-1]
            print(f"      Latest update ID: {latest.update_id}")
            if latest.message:
                print(f"      From: {latest.message.from_user.first_name}")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print("\n" + "="*60)
    print("  TEST COMPLETE")
    print("="*60)
    print("\n✅ All Telegram tests passed!")
    print("   Check your Telegram to see the messages")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_telegram())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
