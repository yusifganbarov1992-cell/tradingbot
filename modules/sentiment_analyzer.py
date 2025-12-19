"""
Sentiment Analysis Module - Phase 6
Анализирует настроения рынка из различных источников:
- Fear & Greed Index (crypto)
- News headlines sentiment
- Social media sentiment (опционально)
"""

import logging
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Уровни настроений рынка"""
    EXTREME_FEAR = "EXTREME_FEAR"      # 0-25
    FEAR = "FEAR"                      # 25-45
    NEUTRAL = "NEUTRAL"                # 45-55
    GREED = "GREED"                    # 55-75
    EXTREME_GREED = "EXTREME_GREED"    # 75-100


class SentimentAnalyzer:
    """
    Анализатор настроений рынка
    
    Собирает данные из:
    1. Crypto Fear & Greed Index (API)
    2. News headlines (опционально)
    3. Social media (опционально - требует API ключи)
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        """
        Args:
            news_api_key: API key для NewsAPI (опционально)
        """
        self.news_api_key = news_api_key
        
        # Fear & Greed Index
        self.fear_greed_url = "https://api.alternative.me/fng/"
        
        # NewsAPI (если есть ключ)
        self.news_api_url = "https://newsapi.org/v2/everything"
        
        # Cache
        self.cached_fear_greed = None
        self.cached_fear_greed_time = None
        self.cache_duration = timedelta(hours=1)
        
        # Current sentiment
        self.current_sentiment = SentimentLevel.NEUTRAL
        self.sentiment_score = 50  # 0-100
        self.sentiment_sources = {}
        
        logger.info("💭 SentimentAnalyzer initialized")
    
    def get_fear_greed_index(self, use_cache: bool = True) -> Dict:
        """
        Получить Crypto Fear & Greed Index
        
        API: https://api.alternative.me/fng/
        
        Returns:
            Dictionary with fear & greed data
        """
        # Check cache
        if use_cache and self.cached_fear_greed and self.cached_fear_greed_time:
            if datetime.now() - self.cached_fear_greed_time < self.cache_duration:
                logger.info("Using cached Fear & Greed Index")
                return self.cached_fear_greed
        
        try:
            # Fetch from API
            params = {'limit': 1}  # Get only latest
            response = requests.get(self.fear_greed_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors (API returns string 'null', not None)
            if 'metadata' not in data or data.get('data') is None:
                logger.error(f"Fear & Greed API invalid response: {data}")
                return {'error': 'Invalid API response'}
            
            # Parse data
            latest = data['data'][0]
            
            result = {
                'value': int(latest['value']),
                'value_classification': latest['value_classification'],
                'timestamp': datetime.fromtimestamp(int(latest['timestamp'])),
                'time_until_update': latest.get('time_until_update', 'N/A')
            }
            
            # Cache result
            self.cached_fear_greed = result
            self.cached_fear_greed_time = datetime.now()
            
            logger.info(f"📊 Fear & Greed Index: {result['value']} ({result['value_classification']})")
            return result
        
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return {'error': str(e)}
    
    def get_fear_greed_history(self, limit: int = 7) -> List[Dict]:
        """
        Получить историю Fear & Greed Index
        
        Args:
            limit: Количество дней
            
        Returns:
            List of historical data
        """
        try:
            params = {'limit': limit}
            response = requests.get(self.fear_greed_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            history = []
            for item in data['data']:
                history.append({
                    'value': int(item['value']),
                    'classification': item['value_classification'],
                    'timestamp': datetime.fromtimestamp(int(item['timestamp']))
                })
            
            return history
        
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed history: {e}")
            return []
    
    def get_news_sentiment(self, query: str = "bitcoin OR cryptocurrency", days: int = 1) -> Dict:
        """
        Получить sentiment из новостей (требует NewsAPI key)
        
        Args:
            query: Search query
            days: Number of days back
            
        Returns:
            Dictionary with news sentiment
        """
        if not self.news_api_key:
            logger.warning("NewsAPI key not provided. Skipping news sentiment.")
            return {'error': 'No API key'}
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'popularity',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(self.news_api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown')}")
                return {'error': 'API error'}
            
            articles = data.get('articles', [])
            
            # Simple sentiment analysis on headlines
            # (В реальном проекте лучше использовать FinBERT или VADER)
            positive_words = ['surge', 'bull', 'rally', 'gain', 'rise', 'high', 'breakthrough', 'adoption', 'positive', 'up']
            negative_words = ['crash', 'bear', 'drop', 'fall', 'decline', 'loss', 'risk', 'concern', 'warning', 'down']
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for article in articles[:20]:  # Analyze top 20
                title = article.get('title', '').lower()
                
                has_positive = any(word in title for word in positive_words)
                has_negative = any(word in title for word in negative_words)
                
                if has_positive and not has_negative:
                    positive_count += 1
                elif has_negative and not has_positive:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = positive_count + negative_count + neutral_count
            
            if total == 0:
                return {'error': 'No articles found'}
            
            # Calculate sentiment score (0-100)
            sentiment_score = ((positive_count - negative_count) / total * 50) + 50
            
            result = {
                'total_articles': len(articles),
                'analyzed': total,
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'sentiment_score': sentiment_score,
                'classification': self._classify_sentiment(sentiment_score)
            }
            
            logger.info(f"📰 News sentiment: {sentiment_score:.1f} ({result['classification']})")
            return result
        
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            return {'error': str(e)}
    
    def _classify_sentiment(self, score: float) -> str:
        """Классифицировать sentiment score в текстовую категорию"""
        if score < 25:
            return "VERY_NEGATIVE"
        elif score < 45:
            return "NEGATIVE"
        elif score < 55:
            return "NEUTRAL"
        elif score < 75:
            return "POSITIVE"
        else:
            return "VERY_POSITIVE"
    
    def get_overall_sentiment(self) -> Dict:
        """
        Получить общий sentiment с учетом всех источников
        
        Веса:
        - Fear & Greed Index: 70%
        - News sentiment: 30%
        
        Returns:
            Dictionary with overall sentiment
        """
        sources = {}
        weights = {}
        
        # 1. Fear & Greed Index (основной источник)
        fear_greed = self.get_fear_greed_index()
        if 'value' in fear_greed:
            sources['fear_greed'] = fear_greed['value']
            weights['fear_greed'] = 0.7
        
        # 2. News sentiment (если доступен)
        if self.news_api_key:
            news = self.get_news_sentiment()
            if 'sentiment_score' in news:
                sources['news'] = news['sentiment_score']
                weights['news'] = 0.3
        
        # Calculate weighted average
        if not sources:
            logger.warning("No sentiment sources available")
            return {
                'overall_score': 50,
                'level': SentimentLevel.NEUTRAL,
                'sources': {},
                'error': 'No data'
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        overall_score = sum(sources[k] * normalized_weights[k] for k in sources.keys())
        
        # Classify
        if overall_score < 25:
            level = SentimentLevel.EXTREME_FEAR
        elif overall_score < 45:
            level = SentimentLevel.FEAR
        elif overall_score < 55:
            level = SentimentLevel.NEUTRAL
        elif overall_score < 75:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED
        
        # Update state
        self.current_sentiment = level
        self.sentiment_score = overall_score
        self.sentiment_sources = sources
        
        result = {
            'overall_score': overall_score,
            'level': level.value,
            'sources': sources,
            'weights': normalized_weights,
            'timestamp': datetime.now()
        }
        
        logger.info(f"💭 Overall sentiment: {overall_score:.1f} ({level.value})")
        return result
    
    def get_trading_recommendation(self) -> Dict:
        """
        Получить рекомендации для торговли на основе sentiment
        
        Returns:
            Dictionary with trading recommendations
        """
        sentiment = self.get_overall_sentiment()
        
        score = sentiment['overall_score']
        level = sentiment['level']
        
        # Рекомендации по sentiment
        recommendations = {
            'EXTREME_FEAR': {
                'action': 'BUY_OPPORTUNITY',
                'description': '🟢 Extreme Fear - хорошая возможность для покупки',
                'confidence_adjustment': -0.5,  # Снизить порог
                'position_size_multiplier': 1.2,  # Увеличить позиции
                'aggressive': True,
                'reasoning': 'Extreme fear часто означает дно рынка'
            },
            'FEAR': {
                'action': 'CAUTIOUS_BUY',
                'description': '🟡 Fear - осторожная покупка',
                'confidence_adjustment': -0.3,
                'position_size_multiplier': 1.1,
                'aggressive': False,
                'reasoning': 'Fear может быть переоценкой'
            },
            'NEUTRAL': {
                'action': 'NORMAL',
                'description': '⚪ Neutral - нормальная торговля',
                'confidence_adjustment': 0.0,
                'position_size_multiplier': 1.0,
                'aggressive': False,
                'reasoning': 'Нейтральный sentiment'
            },
            'GREED': {
                'action': 'CAUTIOUS_SELL',
                'description': '🟠 Greed - осторожно, возможна коррекция',
                'confidence_adjustment': +0.3,
                'position_size_multiplier': 0.9,
                'aggressive': False,
                'reasoning': 'Greed может привести к коррекции'
            },
            'EXTREME_GREED': {
                'action': 'SELL_OPPORTUNITY',
                'description': '🔴 Extreme Greed - высокий риск коррекции',
                'confidence_adjustment': +0.5,  # Повысить порог
                'position_size_multiplier': 0.7,  # Уменьшить позиции
                'aggressive': False,
                'reasoning': 'Extreme greed часто означает пик рынка'
            }
        }
        
        recommendation = recommendations.get(level, recommendations['NEUTRAL'])
        recommendation['sentiment_score'] = score
        recommendation['sentiment_level'] = level
        recommendation['sources'] = sentiment['sources']
        
        return recommendation
    
    def get_sentiment_trend(self, days: int = 7) -> Dict:
        """
        Анализировать тренд sentiment за последние N дней
        
        Args:
            days: Количество дней
            
        Returns:
            Dictionary with trend analysis
        """
        history = self.get_fear_greed_history(limit=days)
        
        if not history:
            return {'error': 'No historical data'}
        
        # Extract values
        values = [item['value'] for item in history]
        
        # Calculate trend
        if len(values) < 2:
            trend = 'STABLE'
        else:
            # Simple linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if slope > 2:
                trend = 'IMPROVING'  # Fear → Greed
            elif slope < -2:
                trend = 'WORSENING'  # Greed → Fear
            else:
                trend = 'STABLE'
        
        # Calculate statistics
        avg_value = np.mean(values)
        volatility = np.std(values)
        
        result = {
            'trend': trend,
            'current': values[0],
            'oldest': values[-1],
            'change': values[0] - values[-1],
            'average': avg_value,
            'volatility': volatility,
            'history': history
        }
        
        logger.info(f"📈 Sentiment trend ({days}d): {trend} (change={result['change']:.1f})")
        return result
    
    def should_adjust_strategy(self) -> Tuple[bool, Dict]:
        """
        Определить, нужно ли корректировать стратегию на основе sentiment
        
        Returns:
            (should_adjust, adjustments)
        """
        recommendation = self.get_trading_recommendation()
        
        # Adjust only on extreme levels
        level = recommendation['sentiment_level']
        
        if level in ['EXTREME_FEAR', 'EXTREME_GREED']:
            return True, {
                'confidence_adjustment': recommendation['confidence_adjustment'],
                'position_size_multiplier': recommendation['position_size_multiplier'],
                'aggressive': recommendation['aggressive'],
                'reason': recommendation['description']
            }
        
        return False, {}
    
    def get_status(self) -> Dict:
        """Получить статус sentiment analyzer"""
        return {
            'current_sentiment': self.current_sentiment.value,
            'sentiment_score': self.sentiment_score,
            'news_api_enabled': self.news_api_key is not None,
            'sources': list(self.sentiment_sources.keys()),
            'cache_valid': (
                self.cached_fear_greed is not None and
                self.cached_fear_greed_time is not None and
                datetime.now() - self.cached_fear_greed_time < self.cache_duration
            )
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create analyzer
    analyzer = SentimentAnalyzer()
    
    # Get Fear & Greed Index
    print("\n1. Fear & Greed Index:")
    fear_greed = analyzer.get_fear_greed_index()
    print(f"   Value: {fear_greed.get('value', 'N/A')}")
    print(f"   Classification: {fear_greed.get('value_classification', 'N/A')}")
    
    # Get overall sentiment
    print("\n2. Overall Sentiment:")
    sentiment = analyzer.get_overall_sentiment()
    print(f"   Score: {sentiment['overall_score']:.1f}")
    print(f"   Level: {sentiment['level']}")
    
    # Get trading recommendation
    print("\n3. Trading Recommendation:")
    recommendation = analyzer.get_trading_recommendation()
    print(f"   Action: {recommendation['action']}")
    print(f"   Description: {recommendation['description']}")
    print(f"   Confidence adjustment: {recommendation['confidence_adjustment']}")
    
    # Get sentiment trend
    print("\n4. Sentiment Trend (7 days):")
    trend = analyzer.get_sentiment_trend(days=7)
    print(f"   Trend: {trend.get('trend', 'N/A')}")
    print(f"   Change: {trend.get('change', 0):.1f}")
    
    # Should adjust strategy?
    print("\n5. Strategy Adjustment:")
    should_adjust, adjustments = analyzer.should_adjust_strategy()
    print(f"   Should adjust: {should_adjust}")
    if should_adjust:
        print(f"   Adjustments: {adjustments}")
