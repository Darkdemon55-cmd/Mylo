# main.py - FastAPI Application Entry Point
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Optional
import asyncio
import os
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import redis.asyncio as redis
from celery import Celery
from pydantic import BaseModel, Field
import json
import logging

# Initialize FastAPI app
app = FastAPI(title="MYLO - Agentic AI Financial Intelligence Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/mylo_db")
engine = create_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Celery setup
celery_app = Celery('mylo_backend')
celery_app.conf.broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
celery_app.conf.result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class UserCreate(BaseModel):
    email: str
    password: str
    tier: str = "FREE"

class UserResponse(BaseModel):
    id: str
    email: str
    tier: str
    role: str
    is_active: bool

class StrategyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    code: str
    language: str = "python"
    tags: List[str] = []

class StrategyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    risk_score: Optional[float]
    acms_approved: bool
    status: str

class AgentRequest(BaseModel):
    query: str
    task_type: str = Field(..., regex="^(quant|coding|research|review|orchestration)$")
    priority: int = Field(0, ge=0, le=2)  # 0=cost, 1=speed, 2=accuracy

class PaymentRequest(BaseModel):
    tier: str = Field(..., regex="^(STARTER|QUANT|ELITE|ENTERPRISE)$")
    amount: float

class RiskLimitUpdate(BaseModel):
    max_drawdown: Optional[float] = Field(None, ge=0.01, le=1.0)
    max_leverage: Optional[float] = Field(None, ge=0.1)
    max_daily_loss: Optional[float] = None

# Dependency for database session
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# JWT token creation
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    # Fetch user from database
    user = await get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

# Routes
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    existing_user = await get_user_by_email(user_data.email, db)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = pwd_context.hash(user_data.password)
    
    # Create user
    user = User(
        email=user_data.email,
        password_hash=hashed_password,
        tier=user_data.tier,
        role="trader"
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    # Create default risk limits
    risk_limit = RiskLimit(
        user_id=user.id,
        max_drawdown=0.25 if user.tier != "FREE" else 0.10,
        max_leverage=1.0 if user.tier != "FREE" else 0.5
    )
    db.add(risk_limit)
    await db.commit()
    
    return UserResponse(
        id=str(user.id),
        email=user.email,
        tier=user.tier,
        role=user.role,
        is_active=user.is_active
    )

@app.post("/auth/login")
async def login_user(credentials: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(credentials.username, credentials.password, db)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/agent/run", response_model=dict)
async def run_agent(
    agent_request: AgentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Check user tier permissions
    if current_user.tier == "FREE" and agent_request.task_type == "quant":
        raise HTTPException(status_code=403, detail="Quant tasks require STARTER tier or higher")
    
    # Queue agent task via Celery
    task = run_agent_task.delay(
        query=agent_request.query,
        task_type=agent_request.task_type,
        user_id=current_user.id,
        priority=agent_request.priority
    )
    
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Agent task submitted successfully"
    }

@app.get("/agent/status/{job_id}")
async def get_agent_status(job_id: str):
    result = celery_app.AsyncResult(job_id)
    return {
        "job_id": job_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }

@app.post("/strategy/create", response_model=StrategyResponse)
async def create_strategy(
    strategy_data: StrategyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Validate code (basic security check)
    if "import os" in strategy_data.code or "exec(" in strategy_data.code:
        raise HTTPException(status_code=400, detail="Strategy contains unsafe operations")
    
    strategy = Strategy(
        user_id=current_user.id,
        name=strategy_data.name,
        description=strategy_data.description,
        code=strategy_data.code,
        language=strategy_data.language,
        tags=strategy_data.tags
    )
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)
    
    return StrategyResponse(
        id=str(strategy.id),
        name=strategy.name,
        description=strategy.description,
        risk_score=strategy.risk_score,
        acms_approved=strategy.acms_approved,
        status=strategy.status
    )

@app.post("/strategy/backtest")
async def run_backtest(
    strategy_id: str,
    parameters: dict = {},
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # Verify user owns strategy
    strategy = await get_strategy_by_id(strategy_id, current_user.id, db)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Queue backtest via Celery
    task = run_backtest_task.delay(
        strategy_id=strategy_id,
        user_id=current_user.id,
        parameters=parameters
    )
    
    return {
        "job_id": task.id,
        "status": "backtest_queued",
        "message": "Backtest submitted successfully"
    }

@app.post("/payment/create-checkout")
async def create_payment_checkout(
    payment_request: PaymentRequest,
    current_user: User = Depends(get_current_user)
):
    # Calculate amount based on tier
    tier_pricing = {
        "STARTER": 29.00,
        "QUANT": 79.00,
        "ELITE": 299.00,
        "ENTERPRISE": 2000.00  # Base amount
    }
    
    amount = tier_pricing.get(payment_request.tier)
    if not amount:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    # In a real implementation, this would call Yoco API
    # For now, return a mock checkout URL
    checkout_url = f"https://yoco-checkout.mock/{current_user.id}/{payment_request.tier}"
    
    # Store pending payment record
    payment_record = {
        "user_id": str(current_user.id),
        "amount": amount,
        "currency": "USD",
        "tier": payment_request.tier,
        "status": "pending",
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Store in Redis temporarily
    await redis_client.setex(f"payment:{current_user.id}", 300, json.dumps(payment_record))
    
    return {
        "checkout_url": checkout_url,
        "session_id": f"mock_session_{current_user.id}",
        "expires_in": 300
    }

@app.put("/user/risk-limits")
async def update_risk_limits(
    risk_update: RiskLimitUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    risk_limit = await get_risk_limit_by_user_id(current_user.id, db)
    if not risk_limit:
        raise HTTPException(status_code=404, detail="Risk limits not found")
    
    # Update allowed fields
    if risk_update.max_drawdown is not None:
        risk_limit.max_drawdown = risk_update.max_drawdown
    if risk_update.max_leverage is not None:
        risk_limit.max_leverage = risk_update.max_leverage
    if risk_update.max_daily_loss is not None:
        risk_limit.max_daily_loss = risk_update.max_daily_loss
    
    await db.commit()
    
    return {"message": "Risk limits updated successfully"}

# Background Tasks
@celery_app.task
def run_agent_task(query: str, task_type: str, user_id: str, priority: int):
    """
    Celery task to run agent orchestration
    This would integrate with the agent system in a real implementation
    """
    logger.info(f"Running agent task: {task_type} for user {user_id}")
    
    # Simulate agent processing
    import time
    time.sleep(2)  # Simulate processing time
    
    # In real implementation:
    # 1. Route to appropriate agent based on task_type
    # 2. Handle multi-model debate if needed
    # 3. Validate output through review agent
    # 4. Return structured response
    
    return {
        "status": "completed",
        "result": f"Processed query: {query[:50]}...",
        "task_type": task_type,
        "user_id": user_id
    }

@celery_app.task
def run_backtest_task(strategy_id: str, user_id: str, parameters: dict):
    """
    Celery task to run backtesting
    This would integrate with the quant engine in a real implementation
    """
    logger.info(f"Running backtest for strategy {strategy_id}")
    
    # Simulate backtest processing
    import time
    time.sleep(3)  # Simulate processing time
    
    # In real implementation:
    # 1. Load strategy code
    # 2. Execute in sandboxed environment
    # 3. Calculate metrics
    # 4. Store results
    
    return {
        "status": "completed",
        "strategy_id": strategy_id,
        "metrics": {
            "sharpe_ratio": 1.25,
            "max_drawdown": 0.15,
            "total_return": 0.28,
            "win_rate": 0.62
        },
        "parameters": parameters
    }

# Utility functions
async def get_user_by_email(email: str, db: AsyncSession):
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalar_one_or_none()

async def authenticate_user(email: str, password: str, db: AsyncSession):
    user = await get_user_by_email(email, db)
    if not user or not pwd_context.verify(password, user.password_hash):
        return False
    return user

async def get_strategy_by_id(strategy_id: str, user_id: str, db: AsyncSession):
    result = await db.execute(
        select(Strategy).filter(Strategy.id == strategy_id, Strategy.user_id == user_id)
    )
    return result.scalar_one_or_none()

async def get_risk_limit_by_user_id(user_id: str, db: AsyncSession):
    result = await db.execute(select(RiskLimit).filter(RiskLimit.user_id == user_id))
    return result.scalar_one_or_none()

# Import SQLAlchemy models
from sqlalchemy import select
from models import User, Strategy, RiskLimit

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
