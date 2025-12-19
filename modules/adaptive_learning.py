"""
Adaptive Learning Module - Phase 4
Использует Reinforcement Learning (PPO) для адаптации параметров торговли
на основе исторических результатов и текущего состояния рынка.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sqlite3

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Gymnasium Environment для обучения RL агента
    Состояние: market metrics, текущие параметры, recent performance
    Действие: корректировка параметров (confidence, stop_loss, take_profit, etc.)
    Награда: win_rate + roi + sharpe_ratio - drawdown
    """
    
    def __init__(self, db_path: str, lookback_days: int = 30):
        super(TradingEnv, self).__init__()
        
        self.db_path = db_path
        self.lookback_days = lookback_days
        
        # Observation space: 15 features
        # [0-4]: Recent performance (win_rate, roi, sharpe, drawdown, avg_pnl)
        # [5-9]: Market state (volatility, trend, volume_ratio, rsi_avg, momentum)
        # [10-14]: Current parameters (confidence, stop_loss, take_profit, position_size, aggressive)
        self.observation_space = spaces.Box(
            low=np.array([0, -100, -5, 0, -100,  # Performance
                         0, -1, 0, 0, -1,          # Market state
                         0, 0, 0, 0, 0], dtype=np.float32),  # Parameters
            high=np.array([100, 100, 5, 100, 100,  # Performance
                          100, 1, 10, 100, 1,      # Market state
                          10, 20, 50, 100, 1], dtype=np.float32),  # Parameters
            dtype=np.float32
        )
        
        # Action space: 5 continuous actions [-1, 1]
        # [0]: Adjust MIN_CONFIDENCE (-1 = decrease, +1 = increase)
        # [1]: Adjust STOP_LOSS_PCT
        # [2]: Adjust TAKE_PROFIT_PCT
        # [3]: Adjust POSITION_SIZE_PCT
        # [4]: Toggle aggressive mode (-1 = conservative, +1 = aggressive)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Current parameters (defaults from config)
        self.current_params = {
            'min_confidence': 7.5,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0,
            'position_size_pct': 5.0,
            'aggressive': 0  # 0 = conservative, 1 = aggressive
        }
        
        # Performance tracking
        self.episode_trades = []
        self.episode_count = 0
        
    def _get_trades_data(self, days: int = None) -> pd.DataFrame:
        """Получить данные сделок из БД"""
        if days is None:
            days = self.lookback_days
            
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT * FROM trades 
                WHERE status = 'closed' 
                AND exit_time >= datetime('now', '-{days} days')
                ORDER BY exit_time ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error loading trades data: {e}")
            return pd.DataFrame()
    
    def _calculate_performance(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Расчет метрик производительности"""
        if len(trades_df) == 0:
            return {
                'win_rate': 0,
                'roi': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_pnl': 0
            }
        
        # Win rate
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
        
        # ROI
        total_pnl = trades_df['pnl'].sum()
        total_invested = trades_df['usdt_amount'].sum()
        roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        # Sharpe ratio
        returns = trades_df['pnl_percent'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        # Avg PnL
        avg_pnl = trades_df['pnl'].mean()
        
        return {
            'win_rate': win_rate,
            'roi': roi,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_pnl': avg_pnl
        }
    
    def _calculate_market_state(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """Расчет состояния рынка"""
        if len(trades_df) == 0:
            return {
                'volatility': 0,
                'trend': 0,
                'volume_ratio': 1,
                'rsi_avg': 50,
                'momentum': 0
            }
        
        # Volatility (std of returns)
        volatility = trades_df['pnl_percent'].std() if len(trades_df) > 1 else 0
        
        # Trend (positive if more wins, negative if more losses)
        recent_trades = trades_df.tail(10)
        wins = len(recent_trades[recent_trades['pnl'] > 0])
        trend = (wins / len(recent_trades) - 0.5) * 2 if len(recent_trades) > 0 else 0
        
        # Volume ratio (mock - would need real market data)
        volume_ratio = 1.0
        
        # RSI average (mock - would need real market data)
        rsi_avg = 50.0
        
        # Momentum (recent performance vs overall)
        recent_roi = self._calculate_performance(trades_df.tail(10))['roi']
        overall_roi = self._calculate_performance(trades_df)['roi']
        momentum = (recent_roi - overall_roi) / 100 if overall_roi != 0 else 0
        
        return {
            'volatility': min(volatility, 100),
            'trend': np.clip(trend, -1, 1),
            'volume_ratio': volume_ratio,
            'rsi_avg': rsi_avg,
            'momentum': np.clip(momentum, -1, 1)
        }
    
    def _get_observation(self) -> np.ndarray:
        """Получить текущее состояние окружения"""
        # Get recent trades
        trades_df = self._get_trades_data()
        
        # Calculate performance
        perf = self._calculate_performance(trades_df)
        
        # Calculate market state
        market = self._calculate_market_state(trades_df)
        
        # Construct observation
        obs = np.array([
            # Performance [0-4]
            perf['win_rate'],
            perf['roi'],
            perf['sharpe_ratio'],
            perf['max_drawdown'],
            perf['avg_pnl'],
            
            # Market state [5-9]
            market['volatility'],
            market['trend'],
            market['volume_ratio'],
            market['rsi_avg'],
            market['momentum'],
            
            # Current parameters [10-14]
            self.current_params['min_confidence'],
            self.current_params['stop_loss_pct'],
            self.current_params['take_profit_pct'],
            self.current_params['position_size_pct'],
            self.current_params['aggressive']
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, trades_df: pd.DataFrame) -> float:
        """
        Расчет награды:
        - Win rate > 50%: +1 за каждый %
        - ROI: +0.1 за каждый %
        - Sharpe ratio > 1: +10
        - Max drawdown: -1 за каждый %
        """
        perf = self._calculate_performance(trades_df)
        
        reward = 0.0
        
        # Win rate reward
        if perf['win_rate'] > 50:
            reward += (perf['win_rate'] - 50) * 1.0
        else:
            reward -= (50 - perf['win_rate']) * 0.5  # Penalty for low win rate
        
        # ROI reward
        reward += perf['roi'] * 0.1
        
        # Sharpe ratio reward
        if perf['sharpe_ratio'] > 1:
            reward += 10
        elif perf['sharpe_ratio'] < 0:
            reward -= 5
        
        # Drawdown penalty
        reward -= perf['max_drawdown'] * 0.1
        
        # Avg PnL reward
        reward += perf['avg_pnl'] * 0.5
        
        return reward
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Сброс окружения"""
        super().reset(seed=seed)
        
        # Reset parameters to defaults
        self.current_params = {
            'min_confidence': 7.5,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0,
            'position_size_pct': 5.0,
            'aggressive': 0
        }
        
        self.episode_trades = []
        self.episode_count += 1
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Шаг окружения"""
        # Apply action to parameters
        self.current_params['min_confidence'] = np.clip(
            self.current_params['min_confidence'] + action[0] * 0.5,  # +/- 0.5
            6.0, 9.5
        )
        
        self.current_params['stop_loss_pct'] = np.clip(
            self.current_params['stop_loss_pct'] + action[1] * 0.5,  # +/- 0.5%
            1.0, 10.0
        )
        
        self.current_params['take_profit_pct'] = np.clip(
            self.current_params['take_profit_pct'] + action[2] * 1.0,  # +/- 1.0%
            3.0, 20.0
        )
        
        self.current_params['position_size_pct'] = np.clip(
            self.current_params['position_size_pct'] + action[3] * 1.0,  # +/- 1.0%
            2.0, 10.0
        )
        
        self.current_params['aggressive'] = 1 if action[4] > 0 else 0
        
        # Get trades with current parameters (simulated)
        trades_df = self._get_trades_data()
        
        # Calculate reward
        reward = self._calculate_reward(trades_df)
        
        # Get new observation
        obs = self._get_observation()
        
        # Episode ends after 100 steps or no trades
        terminated = len(trades_df) == 0
        truncated = False
        
        info = {
            'params': self.current_params.copy(),
            'performance': self._calculate_performance(trades_df)
        }
        
        return obs, reward, terminated, truncated, info


class TrainingCallback(BaseCallback):
    """Callback для логирования прогресса обучения"""
    
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Log every 100 steps
        if self.n_calls % 100 == 0:
            logger.info(f"Training step {self.n_calls}: reward={self.locals.get('rewards', [0])[0]:.2f}")
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        if 'episode_rewards' in self.locals:
            mean_reward = np.mean(self.locals['episode_rewards'])
            self.episode_rewards.append(mean_reward)
            logger.info(f"Episode finished: mean_reward={mean_reward:.2f}")


class AdaptiveLearning:
    """
    Adaptive Learning Manager
    Использует PPO для оптимизации параметров торговли
    """
    
    def __init__(self, db_path: str, model_path: str = "models/adaptive_ppo.zip"):
        self.db_path = db_path
        self.model_path = model_path
        self.env = None
        self.model = None
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load existing model if available
        if os.path.exists(model_path):
            self.load_model()
        
        logger.info(f"🧠 AdaptiveLearning initialized (Trained: {self.is_trained})")
    
    def create_environment(self) -> gym.Env:
        """Создать Gymnasium окружение"""
        env = TradingEnv(db_path=self.db_path)
        return DummyVecEnv([lambda: env])
    
    def train(self, total_timesteps: int = 10000, verbose: int = 1) -> Dict[str, Any]:
        """
        Обучить PPO модель
        
        Args:
            total_timesteps: Количество шагов обучения
            verbose: Уровень логирования
            
        Returns:
            Статистика обучения
        """
        logger.info(f"🧠 Starting PPO training for {total_timesteps} timesteps...")
        
        # Create environment
        self.env = self.create_environment()
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=verbose,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./tensorboard_logs/"
        )
        
        # Train with callback
        callback = TrainingCallback()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save model
        self.model.save(self.model_path)
        self.is_trained = True
        
        logger.info(f"✅ Training complete! Model saved to {self.model_path}")
        
        return {
            'total_timesteps': total_timesteps,
            'episode_rewards': callback.episode_rewards,
            'mean_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0
        }
    
    def load_model(self) -> bool:
        """Загрузить обученную модель"""
        try:
            self.env = self.create_environment()
            self.model = PPO.load(self.model_path, env=self.env)
            self.is_trained = True
            logger.info(f"✅ Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_trained = False
            return False
    
    def predict_optimal_parameters(self) -> Dict[str, float]:
        """
        Предсказать оптимальные параметры на основе текущего состояния
        
        Returns:
            Словарь с рекомендуемыми параметрами
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Using default parameters.")
            return {
                'min_confidence': 7.5,
                'stop_loss_pct': 3.0,
                'take_profit_pct': 6.0,
                'position_size_pct': 5.0,
                'aggressive': False
            }
        
        # Get current observation
        obs = self.env.envs[0]._get_observation()
        
        # Predict action
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Get current params from env
        current_params = self.env.envs[0].current_params
        
        # Apply predicted action
        optimal_params = {
            'min_confidence': float(np.clip(
                current_params['min_confidence'] + action[0] * 0.5,
                6.0, 9.5
            )),
            'stop_loss_pct': float(np.clip(
                current_params['stop_loss_pct'] + action[1] * 0.5,
                1.0, 10.0
            )),
            'take_profit_pct': float(np.clip(
                current_params['take_profit_pct'] + action[2] * 1.0,
                3.0, 20.0
            )),
            'position_size_pct': float(np.clip(
                current_params['position_size_pct'] + action[3] * 1.0,
                2.0, 10.0
            )),
            'aggressive': bool(action[4] > 0)
        }
        
        logger.info(f"🧠 Predicted optimal parameters: {optimal_params}")
        return optimal_params
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Оценить производительность модели
        
        Args:
            n_episodes: Количество эпизодов для оценки
            
        Returns:
            Статистика оценки
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Cannot evaluate.")
            return {'error': 'Model not trained'}
        
        logger.info(f"🧪 Evaluating model for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Safety: max 100 steps per episode
                if episode_length >= 100:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        results = {
            'n_episodes': n_episodes,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'episode_rewards': [float(r) for r in episode_rewards]
        }
        
        logger.info(f"✅ Evaluation complete: mean_reward={results['mean_reward']:.2f}")
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус адаптивного обучения"""
        return {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'model_exists': os.path.exists(self.model_path),
            'env_created': self.env is not None,
            'model_loaded': self.model is not None
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create adaptive learning manager
    adaptive = AdaptiveLearning(db_path="trading_history.db")
    
    # Train model
    stats = adaptive.train(total_timesteps=5000)
    print(f"Training stats: {stats}")
    
    # Predict optimal parameters
    params = adaptive.predict_optimal_parameters()
    print(f"Optimal parameters: {params}")
    
    # Evaluate model
    eval_results = adaptive.evaluate(n_episodes=5)
    print(f"Evaluation: {eval_results}")
