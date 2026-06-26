# trading_engine.py - Quantitative Trading Engine for MYLO Platform
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import vectorbt as vbt
import backtrader as bt
import yfinance as yf
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class Environment(Enum):
    SANDBOX = "sandbox"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class Order:
    symbol: str
    quantity: float
    order_type: OrderType
    side: PositionSide
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: str = field(default_factory=lambda: f"order_{datetime.now().timestamp()}")

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    side: PositionSide
    opened_at: datetime
    unrealized_pnl: float = 0.0
    realied_pnl: float = 0.0
    fees_paid: float = 0.0

@dataclass
class RiskMetrics:
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    volatility: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    alpha: float
    beta: float

@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    metrics: RiskMetrics
    trade_log: List[Dict[str, Any]]
    equity_curve: List[Tuple[datetime, float]]

class TradingEnvironment:
    """Abstract base class for different trading environments"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.commission_rate = commission_rate
        self.current_prices: Dict[str, float] = {}
    
    async def place_order(self, order: Order) -> bool:
        raise NotImplementedError
    
    async def update_positions(self):
        raise NotImplementedError
    
    async def get_account_value(self) -> float:
        raise NotImplementedError

class SandboxEnvironment(TradingEnvironment):
    """Sandbox environment for strategy testing"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        super().__init__(initial_capital, commission_rate)
        self.data_cache: Dict[str, pd.DataFrame] = {}
    
    async def place_order(self, order: Order) -> bool:
        # Validate order
        if order.quantity <= 0:
            return False
        
        # Get current price
        current_price = await self._get_current_price(order.symbol)
        if current_price is None:
            return False
        
        # Calculate cost
        cost = order.quantity * current_price
        commission = cost * self.commission_rate
        
        # Check if we have enough balance for long orders
        if order.side == PositionSide.LONG and cost + commission > self.balance:
            return False
        
        # Execute order
        self.order_history.append(order)
        
        # Update position
        if order.symbol in self.positions:
            pos = self.positions[order.symbol]
            if pos.side == order.side:
                # Same side - increase position
                total_qty = pos.quantity + order.quantity
                avg_price = ((pos.quantity * pos.avg_price) + (order.quantity * current_price)) / total_qty
                pos.quantity = total_qty
                pos.avg_price = avg_price
            else:
                # Opposite side - close existing position
                pnl = (current_price - pos.avg_price) * pos.quantity
                if pos.side == PositionSide.SHORT:
                    pnl = -pnl
                pos.realied_pnl += pnl
                pos.quantity = 0
                # Then open new position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=current_price,
                    side=order.side,
                    opened_at=datetime.now()
                )
        else:
            # New position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                quantity=order.quantity,
                avg_price=current_price,
                side=order.side,
                opened_at=datetime.now()
            )
        
        # Update balance
        self.balance -= (cost + commission)
        
        # Log trade
        self.trade_history.append({
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': current_price,
            'side': order.side.value,
            'timestamp': order.timestamp,
            'commission': commission
        })
        
        return True
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        if symbol not in self.data_cache:
            # Fetch from yfinance
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    self.data_cache[symbol] = hist
                    return hist['Close'].iloc[-1]
            except:
                return None
        else:
            return self.data_cache[symbol]['Close'].iloc[-1]
    
    async def update_positions(self):
        for symbol, pos in self.positions.items():
            current_price = await self._get_current_price(symbol)
            if current_price is not None:
                if pos.side == PositionSide.LONG:
                    pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.avg_price - current_price) * pos.quantity
    
    async def get_account_value(self) -> float:
        total_value = self.balance
        for symbol, pos in self.positions.items():
            current_price = await self._get_current_price(symbol)
            if current_price is not None:
                if pos.side == PositionSide.LONG:
                    total_value += pos.quantity * current_price
                else:
                    total_value += pos.quantity * (2 * pos.avg_price - current_price)
        return total_value

class BacktestingEngine:
    """Advanced backtesting engine with vectorbt and custom analysis"""
    
    def __init__(self):
        self.data_cache = {}
    
    async def run_vectorbt_backtest(
        self,
        strategy_func,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
        **kwargs
    ) -> BacktestResult:
        """Run backtest using vectorbt framework"""
        
        # Fetch data
        if symbol not in self.data_cache:
            data = yf.download(symbol, start=start_date, end=end_date)
            self.data_cache[symbol] = data
        else:
            data = self.data_cache[symbol]
        
        # Generate signals using strategy function
        signals = strategy_func(data)
        
        # Create portfolio
        pf = vbt.Portfolio.from_signals(
            data['Close'],
            entries=signals['entries'],
            exits=signals['exits'],
            init_cash=initial_capital
        )
        
        # Calculate metrics
        total_return = float(pf.total_return())
        sharpe_ratio = float(pf.sharpe_ratio())
        max_drawdown = float(pf.max_drawdown())
        volatility = float(pf.returns().std() * np.sqrt(252))  # Annualized
        
        # Trade statistics
        trade_records = pf.closed_trade_records.to_dict()
        total_trades = len(trade_records['size'])
        winning_trades = sum(1 for r in trade_records['return'] if r > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Value at Risk (95%)
        returns = pf.returns().dropna().values
        if len(returns) > 0:
            var_95 = np.percentile(returns, 5)
        else:
            var_95 = 0.0
        
        # Profit factor
        gross_profit = sum(max(0, r) for r in trade_records['return']) if 'return' in trade_records else 0
        gross_loss = abs(sum(min(0, r) for r in trade_records['return'])) if 'return' in trade_records else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
        
        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Alpha and Beta calculation
        benchmark_returns = vbt.Portfolio.from_holding(data['Close']).returns().dropna()
        strategy_returns = pf.returns().dropna()
        
        if len(benchmark_returns) > 1 and len(strategy_returns) > 1:
            min_len = min(len(benchmark_returns), len(strategy_returns))
            strategy_ret = strategy_returns.iloc[-min_len:].values
            bench_ret = benchmark_returns.iloc[-min_len:].values
            
            # Beta calculation
            covariance_matrix = np.cov(strategy_ret, bench_ret)
            beta = covariance_matrix[0, 1] / covariance_matrix[1, 1] if covariance_matrix[1, 1] != 0 else 0
            
            # Alpha calculation
            benchmark_return = np.mean(bench_ret)
            strategy_expected_return = np.mean(strategy_ret)
            alpha = strategy_expected_return - (0.02 + beta * (benchmark_return - 0.02))  # Assuming 2% risk-free rate
        else:
            alpha, beta = 0.0, 0.0
        
        metrics = RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            alpha=alpha,
            beta=beta
        )
        
        # Generate trade log
        trade_log = []
        for i in range(total_trades):
            trade_log.append({
                'entry_date': str(trade_records['entry_idx'][i]),
                'exit_date': str(trade_records['exit_idx'][i]),
                'entry_price': float(trade_records['entry_price'][i]),
                'exit_price': float(trade_records['exit_price'][i]),
                'size': float(trade_records['size'][i]),
                'return': float(trade_records['return'][i]),
                'pnl': float(trade_records['pnl'][i])
            })
        
        # Equity curve
        equity_curve = [(idx, val) for idx, val in zip(pf.cumulative_returns().index, pf.cumulative_returns().values)]
        
        return BacktestResult(
            strategy_name=kwargs.get('name', 'default_strategy'),
            start_date=datetime.strptime(start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(end_date, '%Y-%m-%d'),
            initial_capital=initial_capital,
            final_value=float(pf.value()),
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            metrics=metrics,
            trade_log=trade_log,
            equity_curve=equity_curve
        )
    
    async def run_monte_carlo_simulation(
        self,
        strategy_returns: List[float],
        num_simulations: int = 10000,
        time_horizon: int = 252  # Trading days in a year
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy validation"""
        
        if len(strategy_returns) < 2:
            return {'error': 'Not enough return data for simulation'}
        
        # Calculate parameters
        mean_return = np.mean(strategy_returns)
        std_deviation = np.std(strategy_returns)
        
        # Generate simulations
        simulations = []
        for _ in range(num_simulations):
            # Simulate returns
            sim_returns = np.random.normal(mean_return, std_deviation, time_horizon)
            # Calculate cumulative returns
            cum_returns = [0]
            for ret in sim_returns:
                cum_returns.append(cum_returns[-1] + ret)
            simulations.append(cum_returns[1:])  # Exclude initial 0
        
        # Calculate percentiles
        p5_returns = np.percentile(simulations, 5, axis=0)
        p50_returns = np.percentile(simulations, 50, axis=0)
        p95_returns = np.percentile(simulations, 95, axis=0)
        
        # Calculate probability metrics
        final_returns = [sim[-1] for sim in simulations]
        prob_positive = sum(1 for r in final_returns if r > 0) / len(final_returns)
        prob_negative = 1 - prob_positive
        
        # VaR metrics
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        
        return {
            'simulations': simulations[:100],  # Return first 100 for visualization
            'percentiles': {
                'p5': p5_returns.tolist(),
                'p50': p50_returns.tolist(),
                'p95': p95_returns.tolist()
            },
            'probabilities': {
                'positive_outcome': prob_positive,
                'negative_outcome': prob_negative
            },
            'var_metrics': {
                'var_95': var_95,
                'var_99': var_99
            },
            'summary': {
                'mean_final_return': np.mean(final_returns),
                'std_final_return': np.std(final_returns),
                'worst_case': np.min(final_returns),
                'best_case': np.max(final_returns)
            }
        }

class RiskManager:
    """Risk management system for trading operations"""
    
    def __init__(self, max_drawdown: float = 0.25, max_leverage: float = 1.0):
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.position_limits = {}
        self.stop_losses = {}
    
    async def validate_order(self, order: Order, account_value: float, positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Validate order against risk parameters"""
        
        # Check if account value is sufficient
        if order.side == PositionSide.LONG:
            # For long orders, check if we have enough cash
            current_price = await self._get_current_price(order.symbol)
            if current_price is None:
                return False, "Unable to get current price"
            
            order_value = order.quantity * current_price
            if order_value > account_value * self.max_leverage:
                return False, f"Order exceeds max leverage ({self.max_leverage})"
        
        # Check if position size is too large
        total_position_value = 0
        for symbol, pos in positions.items():
            current_price = await self._get_current_price(symbol)
            if current_price:
                total_position_value += pos.quantity * current_price
        
        if total_position_value > account_value * 0.9:  # Don't exceed 90% of account value
            return False, "Position size too large"
        
        return True, "Order validated"
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for risk validation"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
        except:
            return None

class StrategyOptimizer:
    """Parameter optimization for trading strategies"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    async def optimize_parameters(
        self,
        strategy_func,
        symbol: str,
        start_date: str,
        end_date: str,
        parameter_space: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using machine learning"""
        
        # Generate parameter combinations
        import itertools
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        combinations = list(itertools.product(*param_values))
        
        # Test each combination
        results = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            try:
                # Run backtest with parameters
                backtest_result = await self._run_single_backtest(
                    strategy_func, symbol, start_date, end_date, params
                )
                
                if backtest_result:
                    metric_value = getattr(backtest_result.metrics, metric, 0)
                    results.append({
                        'params': params,
                        'metric_value': metric_value,
                        'backtest': backtest_result
                    })
            except:
                continue
        
        if not results:
            return {'best_params': {}, 'best_metric': 0, 'all_results': []}
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['metric_value'])
        
        # Train ML model for parameter optimization
        if len(results) > 10:  # Need enough data for training
            X = []
            y = []
            for result in results:
                # Convert parameters to numerical values
                param_vector = []
                for name, value in result['params'].items():
                    if isinstance(value, (int, float)):
                        param_vector.append(value)
                    else:
                        param_vector.append(hash(str(value)) % 10000)
                X.append(param_vector)
                y.append(result['metric_value'])
            
            if X and y:
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
        
        return {
            'best_params': best_result['params'],
            'best_metric': best_result['metric_value'],
            'all_results': sorted(results, key=lambda x: x['metric_value'], reverse=True)[:10],
            'optimal_strategy': best_result['backtest']
        }
    
    async def _run_single_backtest(self, strategy_func, symbol, start_date, end_date, params):
        """Run a single backtest with given parameters"""
        try:
            # Create strategy function with parameters
            def param_strategy(data):
                return strategy_func(data, **params)
            
            engine = BacktestingEngine()
            result = await engine.run_vectorbt_backtest(
                param_strategy, symbol, start_date, end_date
            )
            return result
        except:
            return None

class ExecutionEngine:
    """Main execution engine for live trading"""
    
    def __init__(self, risk_manager: RiskManager, environment: Environment):
        self.risk_manager = risk_manager
        self.environment = environment
        self.active_orders = {}
        self.order_book = {}
    
    async def execute_strategy(
        self,
        strategy_code: str,
        symbol: str,
        initial_capital: float,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a strategy in the specified environment"""
        
        # Execute strategy code in sandbox first
        sandbox_env = SandboxEnvironment(initial_capital)
        
        # Parse and execute strategy
        try:
            # In a real implementation, this would execute in a secure sandbox
            exec_globals = {
                'pd': pd,
                'np': np,
                'vbt': vbt,
                'yf': yf
            }
            
            # Execute strategy logic
            exec(strategy_code, exec_globals)
            
            # Get trading signals from strategy
            # This would depend on the specific strategy implementation
            signals_func = exec_globals.get('generate_signals')
            if signals_func:
                data = yf.download(symbol, period='1mo')  # Recent data
                signals = signals_func(data, **parameters)
                
                # Execute trades based on signals
                for i, signal in enumerate(signals):
                    if signal == 1:  # Buy signal
                        order = Order(
                            symbol=symbol,
                            quantity=parameters.get('position_size', 10),
                            order_type=OrderType.MARKET,
                            side=PositionSide.LONG
                        )
                        success = await sandbox_env.place_order(order)
                        if success:
                            print(f"Executed buy order for {symbol}")
                    elif signal == -1:  # Sell signal
                        order = Order(
                            symbol=symbol,
                            quantity=parameters.get('position_size', 10),
                            order_type=OrderType.MARKET,
                            side=PositionSide.SHORT
                        )
                        success = await sandbox_env.place_order(order)
                        if success:
                            print(f"Executed sell order for {symbol}")
            
            # Return execution results
            account_value = await sandbox_env.get_account_value()
            return {
                'success': True,
                'final_account_value': account_value,
                'trades_executed': len(sandbox_env.trade_history),
                'positions': len(sandbox_env.positions)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

# Example strategy template
def example_momentum_strategy(data, rsi_period=14, rsi_lower=30, rsi_upper=70):
    """
    Example momentum strategy using RSI
    Returns dictionary with 'entries' and 'exits' boolean arrays
    """
    rsi = vbt.RSI.run(data['Close'], window=rsi_period)
    
    entries = (rsi.rsi < rsi_lower) & (rsi.rsi.shift(1) >= rsi_lower)
    exits = (rsi.rsi > rsi_upper) & (rsi.rsi.shift(1) <= rsi_upper)
    
    return {
        'entries': entries,
        'exits': exits
    }

# Example usage
async def main():
    # Initialize engines
    risk_manager = RiskManager(max_drawdown=0.25, max_leverage=2.0)
    execution_engine = ExecutionEngine(risk_manager, Environment.SANDBOX)
    backtest_engine = BacktestingEngine()
    optimizer = StrategyOptimizer()
    
    # Run backtest
    backtest_result = await backtest_engine.run_vectorbt_backtest(
        example_momentum_strategy,
        'BTC-USD',
        '2023-01-01',
        '2023-12-31',
        initial_capital=10000
    )
    
    print(f"Backtest Result: {backtest_result.strategy_name}")
    print(f"Total Return: {backtest_result.total_return:.2%}")
    print(f"Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {backtest_result.max_drawdown:.2%}")
    
    # Run Monte Carlo simulation
    if backtest_result.trade_log:
        returns = [t['return'] for t in backtest_result.trade_log if 'return' in t]
        if returns:
            mc_result = await backtest_engine.run_monte_carlo_simulation(returns[:50])
            print(f"MC Simulation - 95% VaR: {mc_result['var_metrics']['var_95']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
