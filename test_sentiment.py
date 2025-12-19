"""
Тест модуля Sentiment Analysis
Проверяет все функции анализатора настроений
"""

import sys
import logging
from modules.sentiment_analyzer import SentimentAnalyzer, SentimentLevel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_sentiment_analyzer():
    """Тестировать SentimentAnalyzer"""
    
    print("=" * 70)
    print("ТЕСТ SENTIMENT ANALYZER")
    print("=" * 70)
    
    # Create analyzer
    print("\n[1/7] Создание SentimentAnalyzer...")
    analyzer = SentimentAnalyzer()
    print("   ✅ SentimentAnalyzer создан")
    
    # Test Fear & Greed Index
    print("\n[2/7] Получение Fear & Greed Index...")
    fear_greed = analyzer.get_fear_greed_index()
    
    if 'error' in fear_greed:
        print(f"   ❌ Ошибка: {fear_greed['error']}")
    else:
        print(f"   ✅ Fear & Greed Index получен")
        print(f"      Значение: {fear_greed['value']}")
        print(f"      Классификация: {fear_greed['value_classification']}")
        print(f"      Время: {fear_greed['timestamp']}")
    
    # Test Fear & Greed History
    print("\n[3/7] Получение истории Fear & Greed (7 дней)...")
    history = analyzer.get_fear_greed_history(limit=7)
    
    if not history:
        print("   ❌ Не удалось получить историю")
    else:
        print(f"   ✅ История получена ({len(history)} записей)")
        print("      Последние 3 дня:")
        for item in history[:3]:
            print(f"        {item['timestamp'].strftime('%Y-%m-%d')}: {item['value']} ({item['classification']})")
    
    # Test Overall Sentiment
    print("\n[4/7] Расчет общего sentiment...")
    sentiment = analyzer.get_overall_sentiment()
    
    if 'error' in sentiment:
        print(f"   ❌ Ошибка: {sentiment['error']}")
    else:
        print(f"   ✅ Общий sentiment рассчитан")
        print(f"      Score: {sentiment['overall_score']:.1f}/100")
        print(f"      Level: {sentiment['level']}")
        print(f"      Источники: {list(sentiment['sources'].keys())}")
        if 'weights' in sentiment:
            print(f"      Веса: {sentiment['weights']}")
    
    # Test Trading Recommendation
    print("\n[5/7] Получение торговых рекомендаций...")
    recommendation = analyzer.get_trading_recommendation()
    
    print(f"   ✅ Рекомендации получены")
    print(f"      Действие: {recommendation['action']}")
    print(f"      Описание: {recommendation['description']}")
    print(f"      Корректировка confidence: {recommendation['confidence_adjustment']}")
    print(f"      Множитель позиции: {recommendation['position_size_multiplier']}")
    print(f"      Агрессивный режим: {recommendation['aggressive']}")
    print(f"      Обоснование: {recommendation['reasoning']}")
    
    # Test Sentiment Trend
    print("\n[6/7] Анализ тренда sentiment (7 дней)...")
    trend = analyzer.get_sentiment_trend(days=7)
    
    if 'error' in trend:
        print(f"   ❌ Ошибка: {trend['error']}")
    else:
        print(f"   ✅ Тренд проанализирован")
        print(f"      Тренд: {trend['trend']}")
        print(f"      Текущее значение: {trend['current']}")
        print(f"      Старое значение: {trend['oldest']}")
        print(f"      Изменение: {trend['change']:.1f}")
        print(f"      Среднее: {trend['average']:.1f}")
        print(f"      Волатильность: {trend['volatility']:.1f}")
    
    # Test Strategy Adjustment
    print("\n[7/7] Проверка необходимости корректировки стратегии...")
    should_adjust, adjustments = analyzer.should_adjust_strategy()
    
    print(f"   Нужна корректировка: {should_adjust}")
    if should_adjust:
        print(f"   ✅ Корректировки:")
        for key, value in adjustments.items():
            print(f"      {key}: {value}")
    else:
        print(f"   ℹ️ Корректировка не требуется (sentiment в нормальном диапазоне)")
    
    # Get status
    print("\n[СТАТУС] SentimentAnalyzer:")
    status = analyzer.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("ТЕСТ ЗАВЕРШЕН!")
    print("=" * 70)
    
    # Summary
    print("\n📊 РЕЗЮМЕ:")
    print(f"   • Fear & Greed Index работает: {'✅' if 'error' not in fear_greed else '❌'}")
    print(f"   • История доступна: {'✅' if history else '❌'}")
    print(f"   • Общий sentiment: {sentiment.get('level', 'N/A')}")
    print(f"   • Рекомендация: {recommendation['action']}")
    print(f"   • Тренд: {trend.get('trend', 'N/A')}")
    
    return True

if __name__ == "__main__":
    try:
        test_sentiment_analyzer()
    except KeyboardInterrupt:
        print("\n\n❌ Тест прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Ошибка во время теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
