"""
Quick test script for Adaptive Learning (Phase 4)
Tests RL model without running full bot
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.adaptive_learning import AdaptiveLearning

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("ADAPTIVE LEARNING TEST")
    logger.info("=" * 60)
    
    # Create adaptive learning manager
    logger.info("\n[1/5] Initializing AdaptiveLearning...")
    adaptive = AdaptiveLearning(db_path="trading_history.db")
    
    status = adaptive.get_status()
    logger.info(f"Status: {status}")
    
    # Check if model exists
    if status['is_trained']:
        logger.info("✅ Model уже обучена!")
        logger.info("\n[2/5] Skipping training (model exists)")
    else:
        logger.info("⚠️ Model не обучена. Начинаю обучение...")
        logger.info("\n[2/5] Training PPO model (5000 timesteps)...")
        
        try:
            stats = adaptive.train(total_timesteps=5000, verbose=1)
            logger.info(f"✅ Training complete!")
            logger.info(f"   Mean reward: {stats['mean_reward']:.2f}")
            logger.info(f"   Episodes: {len(stats['episode_rewards'])}")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return
    
    # Predict optimal parameters
    logger.info("\n[3/5] Predicting optimal parameters...")
    try:
        params = adaptive.predict_optimal_parameters()
        logger.info("✅ Predicted parameters:")
        for key, value in params.items():
            logger.info(f"   {key}: {value}")
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return
    
    # Evaluate model
    logger.info("\n[4/5] Evaluating model performance...")
    try:
        eval_results = adaptive.evaluate(n_episodes=5)
        logger.info("✅ Evaluation results:")
        logger.info(f"   Mean reward: {eval_results['mean_reward']:.2f}")
        logger.info(f"   Std reward: {eval_results['std_reward']:.2f}")
        logger.info(f"   Mean episode length: {eval_results['mean_length']:.1f}")
        
        if eval_results['mean_reward'] > 0:
            logger.info("   ✅ Model работает хорошо!")
        else:
            logger.info("   ⚠️ Model требует дообучения")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return
    
    # Final status
    logger.info("\n[5/5] Final status check...")
    final_status = adaptive.get_status()
    logger.info(f"✅ AdaptiveLearning готов к использованию!")
    logger.info(f"   Model trained: {final_status['is_trained']}")
    logger.info(f"   Model path: {final_status['model_path']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("ТЕСТ ЗАВЕРШЕН!")
    logger.info("=" * 60)
    logger.info("\nИспользуйте в Telegram боте:")
    logger.info("  /adaptive_status - Статус модели")
    logger.info("  /adaptive_predict - Получить оптимальные параметры")
    logger.info("  /apply_adaptive - Применить параметры")
    logger.info("  /evaluate_adaptive - Оценить производительность")

if __name__ == "__main__":
    main()
