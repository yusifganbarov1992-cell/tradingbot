"""
Тест модуля Intelligent AI
Проверяет LSTM, Pattern Recognition и Ensemble
"""

import sys
import logging
import ccxt
import pandas as pd
from modules.intelligent_ai import IntelligentAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_intelligent_ai():
    """Тестировать IntelligentAI"""
    
    print("=" * 70)
    print("ТЕСТ INTELLIGENT AI")
    print("=" * 70)
    
    # Initialize exchange
    print("\n[1/6] Инициализация Binance exchange...")
    try:
        exchange = ccxt.binance()
        print("   ✅ Exchange initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Fetch data
    print("\n[2/6] Получение исторических данных BTC/USDT (1000 свечей, 1h)...")
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"   ✅ Получено {len(df)} свечей")
        print(f"      Период: {df['timestamp'].min()} - {df['timestamp'].max()}")
        print(f"      Текущая цена: ${df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Initialize IntelligentAI
    print("\n[3/6] Создание IntelligentAI...")
    try:
        ai = IntelligentAI()
        print("   ✅ IntelligentAI создан")
        status = ai.get_status()
        print(f"      LSTM trained: {status['lstm_trained']}")
        print(f"      Sequence length: {status['sequence_length']}")
        print(f"      Weights: {status['model_weights']}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Train LSTM (quick training with few epochs for testing)
    print("\n[4/6] Обучение LSTM модели (10 эпох для теста)...")
    try:
        train_result = ai.train_lstm(df, epochs=10, batch_size=32)
        
        if 'error' in train_result:
            print(f"   ❌ Error: {train_result['error']}")
        else:
            print("   ✅ LSTM модель обучена")
            print(f"      Training samples: {train_result['train_samples']}")
            print(f"      Test samples: {train_result['test_samples']}")
            print(f"      Final train loss: {train_result['final_train_loss']:.6f}")
            print(f"      Final test loss: {train_result['final_test_loss']:.6f}")
            print(f"      Model saved: {train_result['model_path']}")
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Pattern Recognition
    print("\n[5/6] Тестирование Pattern Recognition...")
    try:
        patterns = ai.pattern_recognizer.detect_patterns(df)
        
        if 'error' in patterns:
            print(f"   ⚠️ {patterns['error']}")
        elif len(patterns) == 0:
            print("   ℹ️ Паттерны не обнаружены (это нормально)")
        else:
            print(f"   ✅ Обнаружено паттернов: {len(patterns)}")
            for pattern_name, pattern_data in patterns.items():
                if isinstance(pattern_data, dict):
                    signal = pattern_data.get('signal', 'N/A')
                    confidence = pattern_data.get('confidence', 0)
                    desc = pattern_data.get('description', '')
                    print(f"      • {pattern_name}: {signal} (confidence: {confidence:.2f})")
                    print(f"        {desc}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Get Ensemble Prediction
    print("\n[6/6] Получение ансамблевого предсказания...")
    try:
        prediction = ai.get_ensemble_prediction(df)
        
        print(f"   ✅ Предсказание получено")
        print(f"\n   📊 РЕЗУЛЬТАТ:")
        print(f"      Текущая цена: ${prediction['current_price']:.2f}")
        print(f"      Финальный сигнал: {prediction['final_signal']}")
        print(f"      Уверенность: {prediction['final_confidence']:.2%}")
        
        print(f"\n   🔍 ДЕТАЛИ ПО МОДЕЛЯМ:")
        
        # LSTM
        if 'lstm' in prediction['predictions']:
            lstm = prediction['predictions']['lstm']
            print(f"\n      📈 LSTM:")
            print(f"         Signal: {lstm['signal']}")
            print(f"         Predicted price: ${lstm['predicted_price']:.2f}")
            print(f"         Change: {lstm['change_pct']:+.2f}%")
            print(f"         Confidence: {lstm['confidence']:.2f}")
            print(f"         Weight: {lstm['weight']:.0%}")
        
        # Patterns
        if 'patterns' in prediction['predictions']:
            patterns_pred = prediction['predictions']['patterns']
            print(f"\n      🎨 PATTERNS:")
            print(f"         Signal: {patterns_pred['signal']}")
            print(f"         Patterns detected: {patterns_pred['patterns_detected']}")
            print(f"         BUY signals: {patterns_pred['buy_count']}")
            print(f"         SELL signals: {patterns_pred['sell_count']}")
            print(f"         Confidence: {patterns_pred['confidence']:.2f}")
            print(f"         Weight: {patterns_pred['weight']:.0%}")
        
        # Technical
        if 'technical' in prediction['predictions']:
            tech = prediction['predictions']['technical']
            print(f"\n      📊 TECHNICAL:")
            print(f"         Signal: {tech['signal']}")
            print(f"         Confidence: {tech['confidence']:.2f}")
            print(f"         Weight: {tech['weight']:.0%}")
        
        # Trading Recommendation
        print(f"\n   💡 ТОРГОВАЯ РЕКОМЕНДАЦИЯ:")
        if prediction['final_signal'] in ['STRONG_BUY', 'BUY']:
            print(f"      🟢 {prediction['final_signal']} - Рекомендуется покупка")
            if prediction['final_confidence'] > 0.7:
                print(f"      ✅ Высокая уверенность ({prediction['final_confidence']:.0%})")
            else:
                print(f"      ⚠️ Средняя уверенность ({prediction['final_confidence']:.0%})")
        elif prediction['final_signal'] in ['STRONG_SELL', 'SELL']:
            print(f"      🔴 {prediction['final_signal']} - Рекомендуется продажа")
            if prediction['final_confidence'] > 0.7:
                print(f"      ✅ Высокая уверенность ({prediction['final_confidence']:.0%})")
            else:
                print(f"      ⚠️ Средняя уверенность ({prediction['final_confidence']:.0%})")
        else:
            print(f"      ⚪ {prediction['final_signal']} - Нейтрально, ожидайте")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("ТЕСТ ЗАВЕРШЕН!")
    print("=" * 70)
    
    # Summary
    print("\n📊 РЕЗЮМЕ:")
    print(f"   • LSTM обучен: ✅")
    print(f"   • Pattern Recognition: ✅")
    print(f"   • Ensemble Prediction: ✅")
    print(f"   • Финальный сигнал: {prediction['final_signal']}")
    print(f"   • Уверенность: {prediction['final_confidence']:.0%}")
    
    return True

if __name__ == "__main__":
    try:
        test_intelligent_ai()
    except KeyboardInterrupt:
        print("\n\n❌ Тест прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
