"""
🔄 Вывод USDT из Earn в Spot для торговли
"""
import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

def redeem_usdt_from_earn(amount: float = 50.0):
    """Вывести USDT из Flexible Earn в Spot"""
    import ccxt
    
    print("=" * 50)
    print("🔄 ВЫВОД USDT ИЗ EARN В SPOT")
    print("=" * 50)
    
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_SECRET_KEY'),
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # 1. Проверяем текущий баланс
    print("\n📊 Текущий баланс:")
    balance = exchange.fetch_balance()
    
    spot_usdt = balance.get('USDT', {}).get('free', 0)
    ld_usdt = balance.get('LDUSDT', {}).get('total', 0)
    
    print(f"   Spot USDT: ${spot_usdt:.2f}")
    print(f"   Earn USDT (LDUSDT): ${ld_usdt:.2f}")
    
    if ld_usdt < amount:
        print(f"\n⚠️ Недостаточно в Earn! Доступно: ${ld_usdt:.2f}")
        amount = ld_usdt
        if amount < 1:
            print("❌ Нечего выводить")
            return False
    
    # 2. Получаем productId для USDT
    print(f"\n🔍 Ищем Flexible Earn продукт для USDT...")
    try:
        products = exchange.sapi_get_simple_earn_flexible_list({
            'asset': 'USDT',
            'current': 1,
            'size': 10
        })
        
        if products and 'rows' in products and products['rows']:
            product = products['rows'][0]
            product_id = product['productId']
            print(f"   ✅ Найден: {product_id}")
            print(f"   APY: {product.get('latestAnnualPercentageRate', 'N/A')}")
        else:
            print("   ❌ Продукт не найден, пробуем стандартный ID")
            product_id = 'USDT001'
            
    except Exception as e:
        print(f"   ⚠️ Ошибка получения продукта: {e}")
        product_id = 'USDT001'
    
    # 3. Выводим
    print(f"\n💸 Выводим ${amount:.2f} USDT из Earn...")
    try:
        result = exchange.sapi_post_simple_earn_flexible_redeem({
            'productId': product_id,
            'amount': str(amount),
            'redeemAll': False
        })
        
        print(f"   ✅ Успешно! Result: {result}")
        
        # 4. Проверяем новый баланс
        import time
        time.sleep(2)
        
        new_balance = exchange.fetch_balance()
        new_spot = new_balance.get('USDT', {}).get('free', 0)
        new_earn = new_balance.get('LDUSDT', {}).get('total', 0)
        
        print(f"\n📊 Новый баланс:")
        print(f"   Spot USDT: ${new_spot:.2f} (+${new_spot - spot_usdt:.2f})")
        print(f"   Earn USDT: ${new_earn:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        
        # Попробуем через positions endpoint
        print("\n🔄 Пробуем альтернативный метод...")
        try:
            # Get position
            positions = exchange.sapi_get_simple_earn_flexible_position({
                'asset': 'USDT'
            })
            
            if positions and 'rows' in positions:
                for pos in positions['rows']:
                    print(f"   Position: {pos}")
                    pos_product_id = pos.get('productId')
                    pos_amount = float(pos.get('totalAmount', 0))
                    
                    if pos_amount > 0:
                        redeem_amount = min(amount, pos_amount)
                        result = exchange.sapi_post_simple_earn_flexible_redeem({
                            'productId': pos_product_id,
                            'amount': str(redeem_amount),
                            'redeemAll': False
                        })
                        print(f"   ✅ Redeemed via position: {result}")
                        return True
                        
        except Exception as e2:
            print(f"   ❌ Альтернативный метод тоже не работает: {e2}")
        
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Redeem USDT from Earn to Spot')
    parser.add_argument('--amount', type=float, default=50.0, help='Amount to redeem')
    args = parser.parse_args()
    
    redeem_usdt_from_earn(args.amount)
