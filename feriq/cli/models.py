"""
Model Management for Feriq CLI

Handles LLM model configuration and management including Ollama, OpenAI, and Anthropic.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import click

from .utils import print_error, print_success, print_warning, print_info


class ModelManager:
    """Manages LLM models and providers for the Feriq framework."""
    
    def __init__(self):
        self.providers = {
            'ollama': OllamaProvider(),
            'openai': OpenAIProvider(),
            'anthropic': AnthropicProvider()
        }
    
    def list_all_models(self) -> Dict[str, List[str]]:
        """List all available models from all providers."""
        models = {}
        
        for provider_name, provider in self.providers.items():
            try:
                models[provider_name] = provider.list_models()
            except Exception as e:
                models[provider_name] = []
                if provider.is_available():
                    print_warning(f"Failed to list {provider_name} models: {e}")
        
        return models
    
    def list_ollama_models(self) -> List[str]:
        """List available Ollama models."""
        return self.providers['ollama'].list_models()
    
    def test_model(self, provider: str, model: str) -> bool:
        """Test if a model is working."""
        if provider not in self.providers:
            return False
        
        return self.providers[provider].test_model(model)
    
    def get_model_config(self, provider: str, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        if provider not in self.providers:
            return {}
        
        return self.providers[provider].get_model_config(model)
    
    def validate_provider_setup(self, provider: str) -> bool:
        """Validate that a provider is properly set up."""
        if provider not in self.providers:
            return False
        
        return self.providers[provider].is_available()


class BaseProvider:
    """Base class for model providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        raise NotImplementedError
    
    def list_models(self) -> List[str]:
        """List available models."""
        raise NotImplementedError
    
    def test_model(self, model: str) -> bool:
        """Test if model is working."""
        raise NotImplementedError
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model configuration."""
        raise NotImplementedError


class OllamaProvider(BaseProvider):
    """Ollama model provider."""
    
    def __init__(self):
        super().__init__('ollama')
        self.base_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = []
            for model in response.json().get('models', []):
                models.append(model['name'])
            
            return sorted(models)
        except Exception as e:
            raise Exception(f"Failed to fetch Ollama models: {e}")
    
    def test_model(self, model: str) -> bool:
        """Test Ollama model."""
        try:
            payload = {
                "model": model,
                "prompt": "Hello",
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get Ollama model configuration."""
        return {
            'provider': 'ollama',
            'model': model,
            'base_url': self.base_url,
            'temperature': 0.7,
            'max_tokens': 2048
        }
    
    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            print_info(f"Pulling model {model} from Ollama...")
            
            payload = {"name": model}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'status' in data:
                            print(f"\r{data['status']}", end='', flush=True)
                
                print_success(f"\nModel {model} pulled successfully")
                return True
            else:
                print_error(f"Failed to pull model: {response.text}")
                return False
                
        except Exception as e:
            print_error(f"Error pulling model: {e}")
            return False


class OpenAIProvider(BaseProvider):
    """OpenAI model provider."""
    
    def __init__(self):
        super().__init__('openai')
        self.api_key = os.getenv('OPENAI_API_KEY')
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return bool(self.api_key)
    
    def list_models(self) -> List[str]:
        """List available OpenAI models."""
        if not self.api_key:
            return []
        
        # Common OpenAI models
        return [
            'gpt-4',
            'gpt-4-turbo',
            'gpt-4-turbo-preview',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-16k'
        ]
    
    def test_model(self, model: str) -> bool:
        """Test OpenAI model."""
        if not self.api_key:
            return False
        
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return bool(response.choices)
        except Exception:
            return False
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get OpenAI model configuration."""
        return {
            'provider': 'openai',
            'model': model,
            'api_key': self.api_key,
            'temperature': 0.7,
            'max_tokens': 2048
        }


class AnthropicProvider(BaseProvider):
    """Anthropic model provider."""
    
    def __init__(self):
        super().__init__('anthropic')
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return bool(self.api_key)
    
    def list_models(self) -> List[str]:
        """List available Anthropic models."""
        if not self.api_key:
            return []
        
        return [
            'claude-3-opus-20240229',
            'claude-3-sonnet-20240229',
            'claude-3-haiku-20240307'
        ]
    
    def test_model(self, model: str) -> bool:
        """Test Anthropic model."""
        if not self.api_key:
            return False
        
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return bool(response.content)
        except Exception:
            return False
    
    def get_model_config(self, model: str) -> Dict[str, Any]:
        """Get Anthropic model configuration."""
        return {
            'provider': 'anthropic',
            'model': model,
            'api_key': self.api_key,
            'temperature': 0.7,
            'max_tokens': 2048
        }


def setup_model_interactive():
    """Interactive model setup."""
    manager = ModelManager()
    
    print_info("Setting up model configuration...")
    
    # Check available providers
    available_providers = []
    for provider_name, provider in manager.providers.items():
        if provider.is_available():
            available_providers.append(provider_name)
    
    if not available_providers:
        print_error("No model providers available. Please set up at least one:")
        print("- For Ollama: Install Ollama and ensure it's running")
        print("- For OpenAI: Set OPENAI_API_KEY environment variable")
        print("- For Anthropic: Set ANTHROPIC_API_KEY environment variable")
        return None
    
    print_success(f"Available providers: {', '.join(available_providers)}")
    
    # Select provider
    if len(available_providers) == 1:
        provider = available_providers[0]
        print_info(f"Using {provider} provider")
    else:
        print("\nAvailable providers:")
        for i, provider in enumerate(available_providers, 1):
            print(f"{i}. {provider}")
        
        while True:
            try:
                choice = click.prompt("Select provider", type=int)
                if 1 <= choice <= len(available_providers):
                    provider = available_providers[choice - 1]
                    break
                else:
                    print_error(f"Please enter a number between 1 and {len(available_providers)}")
            except (ValueError, IndexError):
                print_error("Invalid selection. Please enter a number.")
    
    # List models for selected provider
    try:
        models = manager.providers[provider].list_models()
        if not models:
            print_warning(f"No models found for {provider}")
            
            if provider == 'ollama':
                print_info("You can pull models using: feriq model pull <model-name>")
                print_info("Popular models: llama2, mistral, codellama")
            
            return None
        
        print(f"\nAvailable {provider} models:")
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model}")
        
        while True:
            try:
                choice = click.prompt("Select model", type=int)
                if 1 <= choice <= len(models):
                    model = models[choice - 1]
                    break
                else:
                    print_error(f"Please enter a number between 1 and {len(models)}")
            except (ValueError, IndexError):
                print_error("Invalid selection. Please enter a number.")
        
        # Test model
        print_info(f"Testing {provider}:{model}...")
        if manager.test_model(provider, model):
            print_success("Model test successful!")
            return {
                'provider': provider,
                'model': model,
                'config': manager.get_model_config(provider, model)
            }
        else:
            print_error("Model test failed. Please check your configuration.")
            return None
            
    except Exception as e:
        print_error(f"Error setting up model: {e}")
        return None


def list_models_command():
    """List all available models."""
    manager = ModelManager()
    models = manager.list_all_models()
    
    if not any(models.values()):
        print_warning("No models available. Please configure at least one provider.")
        return
    
    for provider, provider_models in models.items():
        if provider_models:
            print_info(f"\n{provider.upper()} Models:")
            for model in provider_models:
                status = "✅" if manager.test_model(provider, model) else "❌"
                print(f"  {status} {model}")
        else:
            print_warning(f"\n{provider.upper()}: Not available or no models")


def pull_ollama_model(model_name: str):
    """Pull an Ollama model."""
    provider = OllamaProvider()
    
    if not provider.is_available():
        print_error("Ollama is not available. Please install and start Ollama.")
        return False
    
    return provider.pull_model(model_name)