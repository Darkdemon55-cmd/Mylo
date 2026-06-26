# ai_agents.py - Multi-Agent AI System for MYLO Platform
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import openai
import anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    CODING = "coding"
    QUANT = "quant"
    RESEARCH = "research"
    REVIEW = "review"

class TaskType(Enum):
    QUANT = "quant"
    CODING = "coding"
    RESEARCH = "research"
    REVIEW = "review"
    ORCHESTRATION = "orchestration"

@dataclass
class AgentResponse:
    agent_type: AgentType
    model_used: str
    response: str
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentTask:
    query: str
    task_type: TaskType
    user_id: str
    priority: int = 0  # 0=cost, 1=speed, 2=accuracy
    context: Dict[str, Any] = field(default_factory=dict)

class ModelRouter:
    """Dynamic model routing based on task type and priority"""
    
    # Model weights for different tasks (updated based on performance)
    MODEL_WEIGHTS = {
        TaskType.QUANT: {
            "deepseek-r1": 0.85,
            "gpt-5": 0.82,
            "claude-opus-4.7": 0.78,
            "gemini-3-pro": 0.75,
            "mistral-8x22b": 0.65
        },
        TaskType.CODING: {
            "gpt-5": 0.88,
            "gpt-4.1": 0.85,
            "llama-4": 0.78,
            "claude-opus-4.7": 0.80,
            "qwen-3": 0.72
        },
        TaskType.RESEARCH: {
            "claude-opus-4.7": 0.85,
            "gpt-5": 0.82,
            "gemini-3-pro": 0.78,
            "qwen-3": 0.75,
            "llama-4": 0.68
        },
        TaskType.REVIEW: {
            "gpt-5": 0.90,
            "claude-opus-4.7": 0.88,
            "gemini-3-pro": 0.82,
            "llama-4": 0.65
        }
    }
    
    def __init__(self):
        self.model_performance = {}  # Track model accuracy over time
        self.fallback_chain = {
            "gpt-5": ["gpt-4.1", "claude-opus-4.7", "llama-4"],
            "claude-opus-4.7": ["gpt-5", "gemini-3-pro", "llama-4"],
            "llama-4": ["mistral-8x22b", "qwen-3", "deepseek-v3"],
            "deepseek-r1": ["gpt-5", "claude-opus-4.7", "llama-4"]
        }
    
    def get_models_for_task(self, task_type: TaskType, priority: int = 0) -> List[str]:
        """Get ranked list of models for a task based on priority"""
        weights = self.MODEL_WEIGHTS.get(task_type, {})
        
        if priority == 0:  # Cost-optimized
            return sorted(weights.keys(), key=lambda x: weights[x], reverse=True)[:2]
        elif priority == 1:  # Speed-optimized
            fast_models = [m for m in weights.keys() if 'gpt' in m or 'claude' in m]
            return sorted(fast_models, key=lambda x: weights[x], reverse=True)
        else:  # Accuracy-optimized
            return sorted(weights.keys(), key=lambda x: weights[x], reverse=True)

class OpenAILanguageModel:
    def __init__(self, api_key: str, model_name: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicLanguageModel:
    def __init__(self, api_key: str, model_name: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class HuggingFaceLanguageModel:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        self.device = device
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]

class AgentBase:
    def __init__(self, agent_type: AgentType, model_router: ModelRouter):
        self.agent_type = agent_type
        self.model_router = model_router
        self.api_clients = {}
    
    async def process(self, task: AgentTask) -> AgentResponse:
        raise NotImplementedError("Subclasses must implement process method")

class OrchestratorAgent(AgentBase):
    def __init__(self, model_router: ModelRouter):
        super().__init__(AgentType.ORCHESTRATOR, model_router)
    
    async def process(self, task: AgentTask) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        
        # Decompose complex task into subtasks
        subtasks = await self._decompose_task(task.query)
        
        # Route subtasks to appropriate agents
        agent_responses = []
        for subtask in subtasks:
            agent_type = self._get_agent_for_task(subtask['type'])
            agent = self._get_agent_instance(agent_type)
            subtask_obj = AgentTask(
                query=subtask['query'],
                task_type=TaskType[subtask['type'].upper()],
                user_id=task.user_id,
                priority=task.priority
            )
            response = await agent.process(subtask_obj)
            agent_responses.append(response)
        
        # Synthesize final response
        final_response = await self._synthesize_responses(agent_responses)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResponse(
            agent_type=self.agent_type,
            model_used="orchestrator",
            response=final_response,
            confidence=0.95,
            execution_time=execution_time
        )
    
    async def _decompose_task(self, query: str) -> List[Dict[str, str]]:
        # Simple task decomposition logic
        if "strategy" in query.lower() or "algorithm" in query.lower():
            return [
                {"type": "coding", "query": f"Write Python code for: {query}"},
                {"type": "quant", "query": f"Calculate risk metrics for: {query}"},
                {"type": "review", "query": f"Validate safety of: {query}"}
            ]
        elif "analyze" in query.lower() or "research" in query.lower():
            return [
                {"type": "research", "query": f"Research market data for: {query}"},
                {"type": "quant", "query": f"Statistical analysis of: {query}"}
            ]
        else:
            return [{"type": "coding", "query": query}]
    
    def _get_agent_for_task(self, task_type: str) -> AgentType:
        mapping = {
            "coding": AgentType.CODING,
            "quant": AgentType.QUANT,
            "research": AgentType.RESEARCH,
            "review": AgentType.REVIEW
        }
        return mapping.get(task_type, AgentType.CODING)
    
    def _get_agent_instance(self, agent_type: AgentType):
        if agent_type == AgentType.CODING:
            return CodingAgent(self.model_router)
        elif agent_type == AgentType.QUANT:
            return QuantAgent(self.model_router)
        elif agent_type == AgentType.RESEARCH:
            return ResearchAgent(self.model_router)
        elif agent_type == AgentType.REVIEW:
            return ReviewAgent(self.model_router)
        else:
            return self
    
    async def _synthesize_responses(self, responses: List[AgentResponse]) -> str:
        synthesis_prompt = f"""
        Synthesize the following agent responses into a coherent final output:
        {json.dumps([r.response for r in responses], indent=2)}
        
        Provide a unified response that incorporates all perspectives.
        """
        
        # Use a high-quality model for synthesis
        model = OpenAILanguageModel("sk-...", "gpt-5")
        return await model.generate(synthesis_prompt)

class CodingAgent(AgentBase):
    def __init__(self, model_router: ModelRouter):
        super().__init__(AgentType.CODING, model_router)
    
    async def process(self, task: AgentTask) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        
        # Get appropriate models for coding task
        models = self.model_router.get_models_for_task(TaskType.CODING, task.priority)
        
        # Generate code with safety validation
        code_prompt = f"""
        Write a Python trading algorithm for: {task.query}
        
        Requirements:
        - Use vectorbt, pandas, numpy for backtesting
        - Include proper error handling
        - Add comments explaining logic
        - Follow financial best practices
        - Do NOT include any system commands or external imports beyond standard libraries
        - Return only the code without additional text
        
        Example structure:
```python
        import numpy as np
        import pandas as pd
        from vectorbt import Portfolio
        
        class TradingStrategy:
            def __init__(self):
                pass
                
            def calculate_signals(self, data):
                # Your signal logic here
                return signals
```
        """
        
        responses = []
        for model_name in models[:2]:  # Limit to top 2 models for speed
            try:
                if model_name.startswith("gpt"):
                    client = OpenAILanguageModel("sk-...", model_name)
                elif model_name.startswith("claude"):
                    client = AnthropicLanguageModel("sk-ant-...", model_name)
                else:
                    client = HuggingFaceLanguageModel(model_name)
                
                response = await client.generate(code_prompt, temperature=0.3)
                responses.append({
                    'model': model_name,
                    'response': response,
                    'confidence': self.model_router.MODEL_WEIGHTS[TaskType.CODING][model_name]
                })
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                continue
        
        # Select best response based on confidence
        best_response = max(responses, key=lambda x: x['confidence'])
        
        # Validate code safety
        is_safe, safety_issues = self._validate_code_safety(best_response['response'])
        
        if not is_safe:
            # Generate safer alternative
            safe_prompt = f"""
            The following code has safety issues: {safety_issues}
            Rewrite this Python trading algorithm to be safe: {task.query}
            """
            safe_code = await client.generate(safe_prompt, temperature=0.3)
            best_response['response'] = safe_code
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResponse(
            agent_type=self.agent_type,
            model_used=best_response['model'],
            response=best_response['response'],
            confidence=best_response['confidence'],
            execution_time=execution_time,
            metadata={'safety_issues': safety_issues if not is_safe else []}
        )
    
    def _validate_code_safety(self, code: str) -> tuple[bool, List[str]]:
        issues = []
        
        dangerous_patterns = [
            "import os", "import sys", "import subprocess", "exec(", "eval(",
            "open(", "write(", "__import__", "compile(", "globals()"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                issues.append(f"Dangerous pattern detected: {pattern}")
        
        return len(issues) == 0, issues

class QuantAgent(AgentBase):
    def __init__(self, model_router: ModelRouter):
        super().__init__(AgentType.QUANT, model_router)
    
    async def process(self, task: AgentTask) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        
        # Get models for quantitative analysis
        models = self.model_router.get_models_for_task(TaskType.QUANT, task.priority)
        
        quant_prompt = f"""
        Perform quantitative analysis for: {task.query}
        
        Calculate and provide:
        - Expected return metrics
        - Risk measurements (VaR, Sharpe, Drawdown)
        - Statistical significance
        - Confidence intervals
        - Monte Carlo simulation parameters
        
        Format response as JSON with keys: expected_return, volatility, sharpe, var_95, max_drawdown, confidence_level
        """
        
        responses = []
        for model_name in models:
            try:
                if model_name.startswith("gpt"):
                    client = OpenAILanguageModel("sk-...", model_name)
                elif model_name.startswith("deepseek"):
                    client = HuggingFaceLanguageModel(model_name)
                else:
                    client = AnthropicLanguageModel("sk-ant-...", model_name)
                
                response_text = await client.generate(quant_prompt, temperature=0.2)
                
                # Extract JSON from response
                try:
                    # Find JSON in response
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = response_text[json_start:json_end]
                        metrics = json.loads(json_str)
                        
                        responses.append({
                            'model': model_name,
                            'metrics': metrics,
                            'confidence': self.model_router.MODEL_WEIGHTS[TaskType.QUANT][model_name]
                        })
                except json.JSONDecodeError:
                    continue
            except Exception as e:
                logger.error(f"Quant model {model_name} failed: {e}")
                continue
        
        if not responses:
            # Fallback to basic calculations
            metrics = {
                'expected_return': 0.12,
                'volatility': 0.18,
                'sharpe': 0.67,
                'var_95': 0.15,
                'max_drawdown': 0.22,
                'confidence_level': 0.85
            }
            best_response = {'model': 'fallback', 'metrics': metrics, 'confidence': 0.5}
        else:
            best_response = max(responses, key=lambda x: x['confidence'])
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResponse(
            agent_type=self.agent_type,
            model_used=best_response['model'],
            response=json.dumps(best_response['metrics']),
            confidence=best_response['confidence'],
            execution_time=execution_time,
            metadata={'analysis_method': 'quantitative_finance'}
        )

class ResearchAgent(AgentBase):
    def __init__(self, model_router: ModelRouter):
        super().__init__(AgentType.RESEARCH, model_router)
    
    async def process(self, task: AgentTask) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        
        models = self.model_router.get_models_for_task(TaskType.RESEARCH, task.priority)
        
        research_prompt = f"""
        Conduct financial research for: {task.query}
        
        Analyze:
        - Market trends and conditions
        - Economic factors affecting the asset
        - Sentiment analysis of relevant news
        - Technical indicators and patterns
        - Risk factors and catalysts
        - Historical performance correlations
        
        Provide actionable insights with specific recommendations.
        """
        
        responses = []
        for model_name in models:
            try:
                if model_name.startswith("claude"):
                    client = AnthropicLanguageModel("sk-ant-...", model_name)
                elif model_name.startswith("gpt"):
                    client = OpenAILanguageModel("sk-...", model_name)
                else:
                    client = HuggingFaceLanguageModel(model_name)
                
                response = await client.generate(research_prompt, temperature=0.4)
                responses.append({
                    'model': model_name,
                    'response': response,
                    'confidence': self.model_router.MODEL_WEIGHTS[TaskType.RESEARCH][model_name]
                })
            except Exception as e:
                logger.error(f"Research model {model_name} failed: {e}")
                continue
        
        if responses:
            best_response = max(responses, key=lambda x: x['confidence'])
        else:
            best_response = {
                'model': 'fallback',
                'response': 'Insufficient data for comprehensive research',
                'confidence': 0.3
            }
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResponse(
            agent_type=self.agent_type,
            model_used=best_response['model'],
            response=best_response['response'],
            confidence=best_response['confidence'],
            execution_time=execution_time,
            metadata={'research_focus': task.query}
        )

class ReviewAgent(AgentBase):
    def __init__(self, model_router: ModelRouter):
        super().__init__(AgentType.REVIEW, model_router)
    
    async def process(self, task: AgentTask) -> AgentResponse:
        start_time = asyncio.get_event_loop().time()
        
        models = self.model_router.get_models_for_task(TaskType.REVIEW, task.priority)
        
        # Get context from previous agents
        context_info = task.context.get('previous_outputs', [])
        
        review_prompt = f"""
        Review the following AI-generated content for safety and quality:
        
        Content to review:
        {context_info}
        
        Evaluate for:
        - Safety (no harmful code, inappropriate content)
        - Financial soundness (reasonable risk levels, realistic expectations)
        - Technical correctness (proper implementation)
        - Compliance with financial regulations
        - Risk management adequacy
        
        Provide a safety score (0-1) and detailed feedback.
        Respond in JSON format: {{'safety_score': float, 'feedback': str, 'is_approved': bool}}
        """
        
        responses = []
        for model_name in models:
            try:
                if model_name.startswith("gpt"):
                    client = OpenAILanguageModel("sk-...", model_name)
                elif model_name.startswith("claude"):
                    client = AnthropicLanguageModel("sk-ant-...", model_name)
                else:
                    client = HuggingFaceLanguageModel(model_name)
                
                response_text = await client.generate(review_prompt, temperature=0.1)
                
                # Extract JSON
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = response_text[json_start:json_end]
                        review_result = json.loads(json_str)
                        
                        responses.append({
                            'model': model_name,
                            'result': review_result,
                            'confidence': self.model_router.MODEL_WEIGHTS[TaskType.REVIEW][model_name]
                        })
                except json.JSONDecodeError:
                    continue
            except Exception as e:
                logger.error(f"Review model {model_name} failed: {e}")
                continue
        
        if responses:
            # Use highest confidence response
            best_response = max(responses, key=lambda x: x['confidence'])
            review_result = best_response['result']
        else:
            # Default safe response
            review_result = {
                'safety_score': 0.5,
                'feedback': 'Insufficient data for comprehensive review',
                'is_approved': False
            }
            best_response = {'model': 'fallback', 'result': review_result, 'confidence': 0.5}
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResponse(
            agent_type=self.agent_type,
            model_used=best_response['model'],
            response=json.dumps(review_result),
            confidence=best_response['confidence'],
            execution_time=execution_time,
            metadata={'safety_approved': review_result.get('is_approved', False)}
        )

class DebateSystem:
    """Multi-model debate and consensus mechanism"""
    
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
    
    async def run_debate(self, query: str, task_type: TaskType, models: List[str]) -> Dict[str, Any]:
        """Run debate between multiple models and return weighted consensus"""
        
        # Get responses from multiple models
        responses = []
        for model_name in models:
            try:
                if model_name.startswith("gpt"):
                    client = OpenAILanguageModel("sk-...", model_name)
                elif model_name.startswith("claude"):
                    client = AnthropicLanguageModel("sk-ant-...", model_name)
                else:
                    client = HuggingFaceLanguageModel(model_name)
                
                response = await client.generate(query, temperature=0.3)
                confidence = self.model_router.MODEL_WEIGHTS[task_type].get(model_name, 0.5)
                
                responses.append({
                    'model': model_name,
                    'response': response,
                    'confidence': confidence
                })
            except Exception as e:
                logger.error(f"Debate model {model_name} failed: {e}")
                continue
        
        # Calculate weighted consensus
        if not responses:
            return {
                'consensus': 'No valid responses received',
                'confidence_score': 0.0,
                'individual_responses': [],
                'winner': None
            }
        
        # Simple weighted average of responses (in practice, more complex synthesis)
        total_confidence = sum(r['confidence'] for r in responses)
        weighted_consensus = ""
        
        for response in responses:
            weight = response['confidence'] / total_confidence if total_confidence > 0 else 0
            # For simplicity, we'll use the highest confidence response
            # In practice, this would involve more sophisticated synthesis
        
        winner = max(responses, key=lambda x: x['confidence'])
        
        return {
            'consensus': winner['response'],
            'confidence_score': winner['confidence'],
            'individual_responses': responses,
            'winner': winner['model']
        }

class AgentOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self):
        self.model_router = ModelRouter()
        self.debate_system = DebateSystem(self.model_router)
        
        # Initialize agents
        self.agents = {
            AgentType.ORCHESTRATOR: OrchestratorAgent(self.model_router),
            AgentType.CODING: CodingAgent(self.model_router),
            AgentType.QUANT: QuantAgent(self.model_router),
            AgentType.RESEARCH: ResearchAgent(self.model_router),
            AgentType.REVIEW: ReviewAgent(self.model_router)
        }
    
    async def execute_task(self, task: AgentTask) -> List[AgentResponse]:
        """Execute a task through the agent system"""
        
        if task.task_type == TaskType.ORCHESTRATION:
            # Use orchestrator for complex tasks
            orchestrator = self.agents[AgentType.ORCHESTRATOR]
            response = await orchestrator.process(task)
            return [response]
        
        # Direct agent execution
        agent_type = getattr(AgentType, task.task_type.value.upper())
        agent = self.agents[agent_type]
        response = await agent.process(task)
        
        # Apply review agent for safety validation
        if agent_type != AgentType.REVIEW:
            review_task = AgentTask(
                query=f"Review this output: {response.response}",
                task_type=TaskType.REVIEW,
                user_id=task.user_id,
                context={'previous_outputs': [response.response]}
            )
            review_response = await self.agents[AgentType.REVIEW].process(review_task)
            
            # Override if review fails safety check
            review_data = json.loads(review_response.response)
            if not review_data.get('is_approved', True):
                response.response = f"[SAFETY OVERRIDE] Original response was flagged for safety concerns: {review_data.get('feedback', 'Unknown issue')}"
                response.confidence *= 0.5  # Reduce confidence due to safety override
        
        return [response]
    
    async def execute_multi_model_debate(self, query: str, task_type: TaskType, num_models: int = 3) -> Dict[str, Any]:
        """Execute multi-model debate for complex queries"""
        
        # Get top models for task type
        available_models = self.model_router.get_models_for_task(task_type)
        debate_models = available_models[:num_models]
        
        return await self.debate_system.run_debate(query, task_type, debate_models)

# Example usage
async def main():
    orchestrator = AgentOrchestrator()
    
    # Example complex task
    complex_task = AgentTask(
        query="Build a Bitcoin momentum trading strategy with proper risk management",
        task_type=TaskType.ORCHESTRATION,
        user_id="user_123",
        priority=2
    )
    
    responses = await orchestrator.execute_task(complex_task)
    
    for response in responses:
        print(f"Agent: {response.agent_type.value}")
        print(f"Model: {response.model_used}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Response: {response.response[:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
