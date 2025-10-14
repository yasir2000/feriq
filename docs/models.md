# Feriq Model Integration Guide

Comprehensive guide to integrating Large Language Models (LLMs) with the Feriq Collaborative AI Agents Framework.

## Table of Contents

1. [Overview](#overview)
2. [Supported Model Providers](#supported-model-providers)
3. [Ollama Integration](#ollama-integration)
4. [OpenAI Integration](#openai-integration)
5. [Anthropic Integration](#anthropic-integration)
6. [Custom Model Providers](#custom-model-providers)
7. [Model Configuration](#model-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Monitoring and Management](#monitoring-and-management)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Overview

Feriq supports multiple LLM providers through a unified interface, allowing you to:

- **Mix and match models** for different agents and tasks
- **Switch providers** without changing your workflow code
- **Optimize costs** by using different models for different complexity levels
- **Ensure reliability** with automatic failover between providers
- **Scale efficiently** with load balancing and caching

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Feriq Framework                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Model Manager                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Provider  │  │    Model    │  │   Request   │  │   │
│  │  │   Registry  │  │   Cache     │  │    Queue    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                               │
├────────────────────────────┼─────────────────────────────────┤
│                            │                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Provider Adapters                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Ollama    │  │   OpenAI    │  │  Anthropic  │  │   │
│  │  │   Adapter   │  │   Adapter   │  │   Adapter   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                               │
├────────────────────────────┼─────────────────────────────────┤
│                            │                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Model Providers                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Ollama    │  │   OpenAI    │  │  Anthropic  │  │   │
│  │  │   Server    │  │     API     │  │     API     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Supported Model Providers

### 1. Ollama (Local Models)

**Advantages:**
- Complete local control and privacy
- No API costs or rate limits
- Offline capability
- Fast inference for small to medium models

**Supported Models:**
- Llama 3.1 (7B, 8B, 70B, 405B)
- Llama 3.2 (1B, 3B, 11B, 90B)
- DeepSeek R1 (1.5B, 7B, 14B, 32B, 67B)
- CodeLlama (7B, 13B, 34B)
- Mistral (7B, 8x7B, 8x22B)
- Phi-3 (3.8B, 14B)
- Gemma (2B, 7B)

### 2. OpenAI

**Advantages:**
- State-of-the-art performance
- Extensive model ecosystem
- Advanced features (function calling, vision)
- Reliable infrastructure

**Supported Models:**
- GPT-4 Turbo
- GPT-4o
- GPT-4o mini
- GPT-3.5 Turbo
- Custom fine-tuned models

### 3. Anthropic

**Advantages:**
- Large context windows
- Excellent reasoning capabilities
- Strong safety features
- Constitutional AI training

**Supported Models:**
- Claude 3.5 Sonnet
- Claude 3 Opus
- Claude 3 Haiku
- Claude Instant

### 4. Other Providers (Coming Soon)

- **Google AI Platform**: Gemini Pro, Gemini Ultra
- **Azure OpenAI**: Enterprise OpenAI models
- **AWS Bedrock**: Multi-provider access
- **Hugging Face**: Community models
- **Cohere**: Command models

## Ollama Integration

### Installation and Setup

#### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows (or download from https://ollama.ai)
winget install Ollama.Ollama
```

#### 2. Start Ollama Service

```bash
# Start Ollama server
ollama serve

# Verify installation
ollama --version
```

#### 3. Pull Models

```bash
# Pull recommended models
ollama pull llama3.1:8b
ollama pull deepseek-r1:1.5b
ollama pull codellama:7b

# List available models
ollama list

# Remove a model
ollama rm llama3.1:8b
```

### Configuration

#### Basic Configuration

```yaml
# feriq.yaml
models:
  default:
    provider: ollama
    model: llama3.1:8b
    base_url: http://localhost:11434
    
  specialized:
    coding:
      provider: ollama
      model: codellama:7b
      temperature: 0.2
      max_tokens: 4096
    
    reasoning:
      provider: ollama
      model: deepseek-r1:1.5b
      temperature: 0.7
      max_tokens: 8192
```

#### Advanced Configuration

```python
from feriq.models import OllamaProvider, ModelConfig

# Configure Ollama provider
ollama_config = ModelConfig(
    provider="ollama",
    base_url="http://localhost:11434",
    timeout=30,
    max_retries=3,
    retry_delay=1.0
)

ollama_provider = OllamaProvider(ollama_config)

# Configure specific models
models = {
    "llama3.1:8b": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 42
    },
    "deepseek-r1:1.5b": {
        "temperature": 0.8,
        "max_tokens": 8192,
        "top_p": 0.95,
        "repeat_penalty": 1.05
    }
}

# Register models
for model_name, params in models.items():
    ollama_provider.register_model(model_name, params)
```

### CLI Integration

```bash
# List available Ollama models
python -m feriq.cli.main model list

# Test a model
python -m feriq.cli.main model test ollama llama3.1:8b

# Pull a new model
python -m feriq.cli.main model pull mistral:7b

# Set default model
python -m feriq.cli.main model set-default ollama llama3.1:8b

# Configure model parameters
python -m feriq.cli.main model configure ollama llama3.1:8b \
  --temperature 0.7 \
  --max-tokens 4096 \
  --top-p 0.9
```

### Model Selection by Task

```python
from feriq import FeriqAgent, ModelSelector

class SmartAgent(FeriqAgent):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.model_selector = ModelSelector()
    
    async def select_model_for_task(self, task: Task) -> str:
        """Select optimal model based on task characteristics."""
        
        if task.category == "coding":
            return "ollama:codellama:7b"
        elif task.complexity == "high" and task.requires_reasoning:
            return "ollama:deepseek-r1:1.5b"
        elif task.length == "long":
            return "ollama:llama3.1:8b"  # Good context window
        else:
            return "ollama:llama3.1:8b"  # Default
    
    async def execute_with_optimal_model(self, task: Task) -> TaskResult:
        """Execute task with automatically selected model."""
        
        model = await self.select_model_for_task(task)
        
        # Temporarily override agent's model
        original_model = self.model_config
        self.model_config = self.model_selector.get_config(model)
        
        try:
            result = await self.execute_task(task)
            return result
        finally:
            # Restore original model
            self.model_config = original_model
```

## OpenAI Integration

### Setup

#### 1. Get API Key

```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

#### 2. Install Dependencies

```bash
pip install openai>=1.0.0
```

### Configuration

```yaml
# feriq.yaml
models:
  openai_default:
    provider: openai
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY}
    organization: ${OPENAI_ORG_ID}  # Optional
    
  openai_fast:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.7
    max_tokens: 4096
    presence_penalty: 0.1
    frequency_penalty: 0.1
```

### Advanced Features

#### Function Calling

```python
from feriq.models import OpenAIProvider
from feriq.tools import FunctionTool

class OpenAIAgent(FeriqAgent):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.provider = OpenAIProvider()
        self.setup_functions()
    
    def setup_functions(self):
        """Setup function calling capabilities."""
        
        # Define available functions
        functions = [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
        
        # Register functions with provider
        for func in functions:
            self.provider.register_function(func["name"], func)
    
    async def execute_with_functions(self, prompt: str) -> Dict[str, Any]:
        """Execute prompt with function calling."""
        
        response = await self.provider.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            functions=self.provider.get_functions(),
            function_call="auto"
        )
        
        # Handle function calls
        if response.function_call:
            function_name = response.function_call.name
            function_args = json.loads(response.function_call.arguments)
            
            # Execute function
            function_result = await self.execute_function(function_name, function_args)
            
            # Send result back to model
            follow_up_response = await self.provider.chat_completion(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "function_call": response.function_call},
                    {"role": "function", "name": function_name, "content": str(function_result)}
                ]
            )
            
            return {
                "response": follow_up_response.content,
                "function_calls": [{
                    "name": function_name,
                    "arguments": function_args,
                    "result": function_result
                }]
            }
        
        return {"response": response.content}
```

#### Vision Capabilities

```python
async def analyze_image(self, image_path: str, prompt: str) -> str:
    """Analyze image using GPT-4 Vision."""
    
    import base64
    
    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    
    response = await self.provider.chat_completion(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    return response.content
```

## Anthropic Integration

### Setup

```bash
# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Install dependencies
pip install anthropic>=0.5.0
```

### Configuration

```yaml
# feriq.yaml
models:
  anthropic_default:
    provider: anthropic
    model: claude-3-5-sonnet-20241022
    api_key: ${ANTHROPIC_API_KEY}
    max_tokens: 8192
    
  anthropic_fast:
    provider: anthropic
    model: claude-3-haiku-20240307
    temperature: 0.7
    max_tokens: 4096
```

### Large Context Windows

```python
class AnthropicAgent(FeriqAgent):
    """Agent optimized for Anthropic's large context windows."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
        self.max_context_length = 200000  # Claude 3.5 Sonnet
    
    async def process_large_document(self, document: str, analysis_prompt: str) -> str:
        """Process large documents using Claude's large context window."""
        
        # Check if document fits in context
        estimated_tokens = len(document) // 4  # Rough estimation
        
        if estimated_tokens > self.max_context_length * 0.8:  # Leave room for response
            # Split document into chunks
            chunks = self.split_document(document, chunk_size=self.max_context_length // 4)
            
            # Process chunks individually
            chunk_results = []
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"""
                Analyze this section (part {i+1} of {len(chunks)}) of a larger document:

                {chunk}

                Analysis request: {analysis_prompt}

                Please provide analysis for this section, noting it's part of a larger document.
                """
                
                result = await self.generate_response(chunk_prompt)
                chunk_results.append(result)
            
            # Synthesize results
            synthesis_prompt = f"""
            I've analyzed {len(chunks)} sections of a document. Here are the individual analyses:

            {chr(10).join(f"Section {i+1}: {result}" for i, result in enumerate(chunk_results))}

            Please synthesize these analyses into a comprehensive overall analysis addressing: {analysis_prompt}
            """
            
            return await self.generate_response(synthesis_prompt)
        
        else:
            # Process entire document at once
            full_prompt = f"""
            Analyze the following document:

            {document}

            Analysis request: {analysis_prompt}
            """
            
            return await self.generate_response(full_prompt)
```

## Custom Model Providers

### Creating a Custom Provider

```python
from feriq.models.base import BaseModelProvider
from typing import Dict, Any, List, AsyncGenerator
import aiohttp

class CustomModelProvider(BaseModelProvider):
    """Custom model provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url")
        self.api_key = config.get("api_key")
        self.session = None
    
    async def initialize(self):
        """Initialize the provider."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
    
    async def generate_response(self, 
                               prompt: str, 
                               model: str = None,
                               **kwargs) -> str:
        """Generate response from the model."""
        
        request_data = {
            "model": model or self.default_model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        async with self.session.post(
            f"{self.base_url}/completions",
            json=request_data
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["text"]
    
    async def stream_response(self, 
                             prompt: str, 
                             model: str = None,
                             **kwargs) -> AsyncGenerator[str, None]:
        """Stream response from the model."""
        
        request_data = {
            "model": model or self.default_model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True
        }
        
        async with self.session.post(
            f"{self.base_url}/completions",
            json=request_data
        ) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if data.get("choices"):
                        text = data["choices"][0].get("text", "")
                        if text:
                            yield text
    
    async def chat_completion(self, 
                             messages: List[Dict[str, str]], 
                             model: str = None,
                             **kwargs) -> str:
        """Chat completion endpoint."""
        
        request_data = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            json=request_data
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models."""
        
        async with self.session.get(f"{self.base_url}/models") as response:
            response.raise_for_status()
            data = await response.json()
            return [model["id"] for model in data["data"]]
    
    async def validate_model(self, model: str) -> bool:
        """Validate if model is available."""
        
        available_models = await self.get_available_models()
        return model in available_models
```

### Registering Custom Providers

```python
from feriq.models import ModelManager

# Register custom provider
model_manager = ModelManager()
model_manager.register_provider("custom", CustomModelProvider)

# Configure custom provider
custom_config = {
    "base_url": "https://api.customprovider.com/v1",
    "api_key": "your-api-key",
    "default_model": "custom-model-v1"
}

await model_manager.add_provider("my_custom", "custom", custom_config)

# Use custom provider
response = await model_manager.generate_response(
    prompt="Hello, world!",
    provider="my_custom",
    model="custom-model-v1"
)
```

## Model Configuration

### Configuration Hierarchy

1. **Agent-level configuration** (highest priority)
2. **Task-level configuration**
3. **Workflow-level configuration**
4. **Project-level configuration**
5. **Global configuration** (lowest priority)

### Dynamic Configuration

```python
from feriq.models import DynamicModelConfig

class AdaptiveAgent(FeriqAgent):
    """Agent that adapts model configuration based on context."""
    
    def __init__(self, name: str):
        super().__init__(name=name)
        self.config_manager = DynamicModelConfig()
    
    async def adapt_config_for_task(self, task: Task) -> Dict[str, Any]:
        """Adapt model configuration for specific task."""
        
        base_config = self.model_config.copy()
        
        # Adjust based on task complexity
        if task.complexity == "high":
            base_config["temperature"] = 0.3  # More focused
            base_config["max_tokens"] = 8192   # More space for reasoning
        elif task.complexity == "creative":
            base_config["temperature"] = 0.9  # More creative
            base_config["top_p"] = 0.95       # More diverse
        
        # Adjust based on task type
        if task.type == "coding":
            base_config["temperature"] = 0.2  # Very focused
            base_config["model"] = "codellama:7b"
        elif task.type == "analysis":
            base_config["model"] = "deepseek-r1:1.5b"
        
        # Adjust based on urgency
        if task.urgency == "high":
            base_config["model"] = "gpt-3.5-turbo"  # Faster model
        
        return base_config
```

### Environment-Specific Configuration

```yaml
# config/development.yaml
models:
  default:
    provider: ollama
    model: llama3.1:8b
    base_url: http://localhost:11434

# config/staging.yaml
models:
  default:
    provider: openai
    model: gpt-3.5-turbo
    api_key: ${OPENAI_API_KEY_STAGING}

# config/production.yaml
models:
  default:
    provider: openai
    model: gpt-4-turbo-preview
    api_key: ${OPENAI_API_KEY_PROD}
  
  fallback:
    provider: anthropic
    model: claude-3-haiku-20240307
    api_key: ${ANTHROPIC_API_KEY}
```

## Performance Optimization

### Caching Strategies

```python
from feriq.models.cache import ModelCache, CacheStrategy

class OptimizedModelManager:
    """Model manager with advanced caching."""
    
    def __init__(self):
        self.cache = ModelCache(
            strategy=CacheStrategy.LRU,
            max_size=1000,
            ttl=3600
        )
        self.semantic_cache = SemanticCache()
    
    async def generate_with_cache(self, 
                                 prompt: str, 
                                 model: str,
                                 **kwargs) -> str:
        """Generate response with caching."""
        
        # Check exact match cache
        cache_key = self.generate_cache_key(prompt, model, kwargs)
        cached_response = await self.cache.get(cache_key)
        
        if cached_response:
            return cached_response
        
        # Check semantic similarity cache
        similar_response = await self.semantic_cache.find_similar(
            prompt, similarity_threshold=0.95
        )
        
        if similar_response:
            return similar_response
        
        # Generate new response
        response = await self.model_provider.generate_response(
            prompt, model, **kwargs
        )
        
        # Cache the response
        await self.cache.set(cache_key, response)
        await self.semantic_cache.add(prompt, response)
        
        return response
```

### Load Balancing

```python
class LoadBalancedModelManager:
    """Model manager with load balancing across providers."""
    
    def __init__(self):
        self.providers = {}
        self.load_tracker = LoadTracker()
        self.health_checker = HealthChecker()
    
    async def select_optimal_provider(self, model_requirements: Dict[str, Any]) -> str:
        """Select the best provider based on load and health."""
        
        # Get healthy providers
        healthy_providers = []
        for provider_name, provider in self.providers.items():
            if await self.health_checker.is_healthy(provider):
                healthy_providers.append(provider_name)
        
        if not healthy_providers:
            raise NoHealthyProvidersException("No healthy providers available")
        
        # Filter by model availability
        compatible_providers = []
        for provider_name in healthy_providers:
            if await self.providers[provider_name].supports_model(
                model_requirements.get("model")
            ):
                compatible_providers.append(provider_name)
        
        if not compatible_providers:
            raise NoCompatibleProvidersException("No providers support requested model")
        
        # Select based on load
        loads = {}
        for provider_name in compatible_providers:
            loads[provider_name] = await self.load_tracker.get_load(provider_name)
        
        # Return provider with lowest load
        return min(loads.items(), key=lambda x: x[1])[0]
```

### Request Batching

```python
class BatchedModelManager:
    """Model manager with request batching."""
    
    def __init__(self):
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms
        self.pending_requests = []
        self.batch_timer = None
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response with automatic batching."""
        
        # Create request
        request = ModelRequest(
            prompt=prompt,
            future=asyncio.Future(),
            **kwargs
        )
        
        # Add to batch
        self.pending_requests.append(request)
        
        # Process batch if full or start timer
        if len(self.pending_requests) >= self.batch_size:
            await self.process_batch()
        elif self.batch_timer is None:
            self.batch_timer = asyncio.create_task(
                self.batch_timeout_handler()
            )
        
        # Wait for response
        return await request.future
    
    async def process_batch(self):
        """Process a batch of requests."""
        
        if not self.pending_requests:
            return
        
        # Cancel timer if running
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Get current batch
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        
        try:
            # Group by model and provider
            grouped_requests = self.group_requests_by_model(batch)
            
            # Process each group
            for (provider, model), requests in grouped_requests.items():
                responses = await self.providers[provider].batch_generate(
                    prompts=[req.prompt for req in requests],
                    model=model,
                    **requests[0].kwargs
                )
                
                # Set results
                for request, response in zip(requests, responses):
                    request.future.set_result(response)
        
        except Exception as e:
            # Set exception for all requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
```

## Monitoring and Management

### Health Monitoring

```python
from feriq.models.monitoring import ModelHealthMonitor

class HealthMonitor:
    """Monitor model provider health and performance."""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = AlertManager()
    
    async def check_provider_health(self, provider: str) -> HealthStatus:
        """Check health of a specific provider."""
        
        start_time = time.time()
        
        try:
            # Test basic functionality
            test_response = await self.providers[provider].generate_response(
                "Hello", timeout=5
            )
            
            response_time = time.time() - start_time
            
            # Update metrics
            self.update_metrics(provider, {
                "response_time": response_time,
                "success_rate": 1.0,
                "last_success": time.time()
            })
            
            # Determine health status
            if response_time > 10:
                return HealthStatus.DEGRADED
            elif response_time > 5:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        
        except Exception as e:
            self.update_metrics(provider, {
                "success_rate": 0.0,
                "last_error": str(e),
                "last_error_time": time.time()
            })
            
            return HealthStatus.UNHEALTHY
    
    async def monitor_continuously(self):
        """Continuously monitor all providers."""
        
        while True:
            for provider_name in self.providers.keys():
                health = await self.check_provider_health(provider_name)
                
                if health in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                    await self.alerts.send_alert(
                        f"Provider {provider_name} health: {health.value}"
                    )
            
            await asyncio.sleep(60)  # Check every minute
```

### Usage Analytics

```python
class ModelUsageAnalytics:
    """Track and analyze model usage patterns."""
    
    def __init__(self):
        self.usage_db = UsageDatabase()
        self.cost_calculator = CostCalculator()
    
    async def track_request(self, 
                           provider: str, 
                           model: str, 
                           request: ModelRequest,
                           response: ModelResponse):
        """Track a model request for analytics."""
        
        usage_record = {
            "timestamp": time.time(),
            "provider": provider,
            "model": model,
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(response.content.split()),
            "response_time": response.response_time,
            "success": response.success,
            "cost": await self.cost_calculator.calculate_cost(
                provider, model, request, response
            )
        }
        
        await self.usage_db.insert(usage_record)
    
    async def generate_usage_report(self, 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, Any]:
        """Generate usage analytics report."""
        
        records = await self.usage_db.query(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "total_requests": len(records),
            "total_cost": sum(r["cost"] for r in records),
            "avg_response_time": sum(r["response_time"] for r in records) / len(records),
            "success_rate": sum(1 for r in records if r["success"]) / len(records),
            "usage_by_provider": self.group_by_provider(records),
            "usage_by_model": self.group_by_model(records),
            "cost_breakdown": self.calculate_cost_breakdown(records)
        }
```

## Best Practices

### 1. Model Selection Guidelines

```python
class ModelSelectionGuide:
    """Guidelines for selecting appropriate models."""
    
    MODEL_RECOMMENDATIONS = {
        "coding": {
            "primary": "codellama:7b",
            "alternatives": ["gpt-4", "claude-3-sonnet"],
            "reasoning": "Specialized for code generation and understanding"
        },
        "reasoning": {
            "primary": "deepseek-r1:1.5b",
            "alternatives": ["gpt-4", "claude-3-opus"],
            "reasoning": "Optimized for complex reasoning tasks"
        },
        "creative_writing": {
            "primary": "claude-3-opus",
            "alternatives": ["gpt-4", "llama3.1:8b"],
            "reasoning": "Excellent at creative and nuanced writing"
        },
        "analysis": {
            "primary": "gpt-4-turbo",
            "alternatives": ["claude-3-sonnet", "deepseek-r1:1.5b"],
            "reasoning": "Strong analytical capabilities with function calling"
        },
        "general": {
            "primary": "llama3.1:8b",
            "alternatives": ["gpt-3.5-turbo", "claude-3-haiku"],
            "reasoning": "Good balance of capability and cost"
        }
    }
    
    @classmethod
    def recommend_model(cls, task_type: str, constraints: Dict[str, Any] = None) -> str:
        """Recommend a model for a given task type."""
        
        constraints = constraints or {}
        
        if task_type not in cls.MODEL_RECOMMENDATIONS:
            task_type = "general"
        
        recommendation = cls.MODEL_RECOMMENDATIONS[task_type]
        
        # Apply constraints
        if constraints.get("local_only"):
            # Filter to local models only
            local_models = ["codellama:7b", "deepseek-r1:1.5b", "llama3.1:8b"]
            models = [recommendation["primary"]] + recommendation["alternatives"]
            
            for model in models:
                if any(local in model for local in local_models):
                    return model
            
            return "llama3.1:8b"  # Default local model
        
        if constraints.get("max_cost"):
            # Consider cost constraints
            cost_tiers = {
                "low": ["llama3.1:8b", "gpt-3.5-turbo", "claude-3-haiku"],
                "medium": ["gpt-4", "claude-3-sonnet"],
                "high": ["gpt-4-turbo", "claude-3-opus"]
            }
            
            max_cost = constraints["max_cost"]
            for tier in ["low", "medium", "high"]:
                if max_cost <= {"low": 0.01, "medium": 0.05, "high": 0.1}[tier]:
                    models = [recommendation["primary"]] + recommendation["alternatives"]
                    for model in models:
                        if model in cost_tiers[tier]:
                            return model
                    break
        
        return recommendation["primary"]
```

### 2. Error Handling and Fallbacks

```python
class RobustModelManager:
    """Model manager with comprehensive error handling."""
    
    def __init__(self):
        self.fallback_chain = [
            ("openai", "gpt-4-turbo"),
            ("anthropic", "claude-3-sonnet"),
            ("ollama", "llama3.1:8b")
        ]
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2
        }
    
    async def generate_with_fallback(self, 
                                    prompt: str, 
                                    preferred_provider: str = None,
                                    **kwargs) -> str:
        """Generate response with automatic fallback."""
        
        # Try preferred provider first
        if preferred_provider:
            try:
                return await self.generate_with_retry(
                    preferred_provider, prompt, **kwargs
                )
            except Exception as e:
                self.logger.warning(f"Preferred provider {preferred_provider} failed: {e}")
        
        # Try fallback chain
        for provider, model in self.fallback_chain:
            try:
                return await self.generate_with_retry(
                    provider, prompt, model=model, **kwargs
                )
            except Exception as e:
                self.logger.warning(f"Fallback provider {provider} failed: {e}")
                continue
        
        raise AllProvidersFailedException("All providers failed")
    
    async def generate_with_retry(self, 
                                 provider: str, 
                                 prompt: str,
                                 **kwargs) -> str:
        """Generate response with retry logic."""
        
        last_exception = None
        
        for attempt in range(self.retry_config["max_retries"] + 1):
            try:
                return await self.providers[provider].generate_response(
                    prompt, **kwargs
                )
            
            except (TimeoutError, ConnectionError, ServerError) as e:
                last_exception = e
                
                if attempt < self.retry_config["max_retries"]:
                    delay = min(
                        self.retry_config["base_delay"] * 
                        (self.retry_config["exponential_base"] ** attempt),
                        self.retry_config["max_delay"]
                    )
                    
                    await asyncio.sleep(delay)
                    continue
                else:
                    break
            
            except (AuthenticationError, QuotaExceededError) as e:
                # Don't retry these errors
                raise e
        
        raise last_exception
```

### 3. Cost Optimization

```python
class CostOptimizer:
    """Optimize model usage for cost efficiency."""
    
    def __init__(self):
        self.cost_per_token = {
            "gpt-4-turbo": {"input": 0.00001, "output": 0.00003},
            "gpt-3.5-turbo": {"input": 0.000001, "output": 0.000002},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
            "ollama": {"input": 0, "output": 0}  # Local models are free
        }
    
    def estimate_cost(self, 
                     prompt: str, 
                     expected_response_length: int,
                     model: str) -> float:
        """Estimate cost for a request."""
        
        if model.startswith("ollama:"):
            return 0.0
        
        if model not in self.cost_per_token:
            model = "gpt-3.5-turbo"  # Default
        
        input_tokens = len(prompt.split())
        output_tokens = expected_response_length
        
        costs = self.cost_per_token[model]
        
        return (input_tokens * costs["input"]) + (output_tokens * costs["output"])
    
    def select_cost_effective_model(self, 
                                   task: Task,
                                   budget: float) -> str:
        """Select most cost-effective model for task."""
        
        # Estimate response length based on task
        response_length = self.estimate_response_length(task)
        
        # Calculate costs for different models
        viable_models = []
        
        for model in self.cost_per_token.keys():
            cost = self.estimate_cost(task.prompt, response_length, model)
            
            if cost <= budget:
                quality_score = self.get_quality_score(model, task.type)
                efficiency = quality_score / max(cost, 0.0001)  # Avoid division by zero
                
                viable_models.append({
                    "model": model,
                    "cost": cost,
                    "quality": quality_score,
                    "efficiency": efficiency
                })
        
        if not viable_models:
            return "ollama:llama3.1:8b"  # Fallback to free local model
        
        # Select model with best efficiency (quality/cost ratio)
        best_model = max(viable_models, key=lambda x: x["efficiency"])
        
        return best_model["model"]
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Ollama Connection Issues

```bash
# Problem: Connection refused to Ollama
# Solution: Ensure Ollama is running
ollama serve

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check Ollama logs
ollama logs

# Restart Ollama if needed
pkill ollama
ollama serve
```

#### 2. Model Not Found

```bash
# Problem: Model 'llama3.1:8b' not found
# Solution: Pull the model
ollama pull llama3.1:8b

# List available models
ollama list

# Check model status
python -m feriq.cli.main model test ollama llama3.1:8b
```

#### 3. API Key Issues

```python
# Problem: Invalid API key
# Solution: Verify API key configuration

import os
from feriq.models import validate_api_keys

# Check environment variables
print("OpenAI Key:", os.getenv("OPENAI_API_KEY", "Not set"))
print("Anthropic Key:", os.getenv("ANTHROPIC_API_KEY", "Not set"))

# Validate API keys
validation_results = await validate_api_keys()
for provider, status in validation_results.items():
    print(f"{provider}: {status}")
```

#### 4. Memory Issues

```python
# Problem: Out of memory with large models
# Solution: Optimize memory usage

# Use smaller models
config = {
    "models": {
        "default": {
            "provider": "ollama",
            "model": "llama3.1:8b"  # Instead of 70b
        }
    }
}

# Limit concurrent requests
semaphore = asyncio.Semaphore(2)  # Only 2 concurrent requests

async def limited_generate(prompt):
    async with semaphore:
        return await model_manager.generate_response(prompt)
```

#### 5. Rate Limiting

```python
# Problem: Rate limit exceeded
# Solution: Implement rate limiting

from feriq.models.rate_limiting import RateLimiter

rate_limiter = RateLimiter(
    requests_per_minute=60,
    burst_size=10
)

async def rate_limited_generate(prompt):
    async with rate_limiter:
        return await model_manager.generate_response(prompt)
```

### Diagnostic Tools

```bash
# Check system status
python -m feriq.cli.main doctor

# Test all providers
python -m feriq.cli.main model test-all

# Validate configuration
python -m feriq.cli.main config validate

# Check model availability
python -m feriq.cli.main model available --provider ollama

# Monitor resource usage
python -m feriq.cli.main monitor --real-time

# View logs
python -m feriq.cli.main logs show --provider ollama --last 50
```

---

*This model integration guide provides comprehensive coverage of LLM integration with Feriq. For CLI usage, see the [CLI User Guide](cli_guide.md), and for programming examples, see the [Programming Guide](programming_guide.md).*