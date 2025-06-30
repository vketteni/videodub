"""OpenAI API adapter."""

import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

import structlog

from ..core.exceptions import APIError, RateLimitError, AuthenticationError, QuotaExceededError
from ..utils.cost_tracking import track_openai_usage, track_tts_usage


logger = structlog.get_logger(__name__)


class OpenAIAdapter:
    """Adapter for OpenAI API interactions."""
    
    def __init__(self, api_key: str):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self._client = None
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Set up the OpenAI client."""
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized")
        except ImportError as e:
            raise APIError(
                "OpenAI package not available. Install with: pip install openai",
                details={"error": str(e)}
            )
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Any:
        """
        Create a chat completion.
        
        Args:
            model: Model to use
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
            
        Raises:
            APIError: If request fails
        """
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Track usage for cost calculation
            if hasattr(response, 'usage') and response.usage:
                usage_data = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                track_openai_usage(model, usage_data)
            
            logger.debug(
                "Chat completion successful",
                model=model,
                messages_count=len(messages),
                max_tokens=max_tokens,
                usage=usage_data if hasattr(response, 'usage') and response.usage else None
            )
            
            return response
            
        except Exception as e:
            logger.error("Chat completion failed", error=str(e))
            self._handle_openai_error(e)
    
    async def text_to_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1",
        response_format: str = "wav",
        output_path: Optional[Path] = None
    ) -> bytes:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            model: TTS model to use
            response_format: Audio format
            output_path: Optional path to save audio file
            
        Returns:
            Audio data as bytes
            
        Raises:
            APIError: If request fails
        """
        try:
            response = await self._client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=response_format
            )
            
            # Track TTS usage for cost calculation
            track_tts_usage(model, len(text))
            
            # Get audio data
            audio_data = response.content
            
            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                
                logger.debug(
                    "TTS audio saved",
                    output_path=str(output_path),
                    text_length=len(text),
                    voice=voice
                )
            
            return audio_data
            
        except Exception as e:
            logger.error("Text-to-speech failed", text_length=len(text), error=str(e))
            self._handle_openai_error(e)
    
    async def get_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of model names
            
        Raises:
            APIError: If request fails
        """
        try:
            response = await self._client.models.list()
            models = [model.id for model in response.data]
            
            logger.debug("Retrieved models list", model_count=len(models))
            return models
            
        except Exception as e:
            logger.error("Failed to get models", error=str(e))
            self._handle_openai_error(e)
    
    def _handle_openai_error(self, error: Exception) -> None:
        """
        Handle OpenAI API errors and convert to appropriate exceptions.
        
        Args:
            error: Original exception
            
        Raises:
            Appropriate APIError subclass
        """
        error_str = str(error).lower()
        
        # Check for specific error types
        if "rate limit" in error_str or "429" in error_str:
            raise RateLimitError("Rate limit exceeded", status_code=429)
        
        if "authentication" in error_str or "401" in error_str:
            raise AuthenticationError("Authentication failed", status_code=401)
        
        if "quota" in error_str or "insufficient" in error_str:
            raise QuotaExceededError("API quota exceeded")
        
        if "timeout" in error_str:
            raise APIError("Request timeout", status_code=408)
        
        # Check for HTTP status codes in error
        if hasattr(error, 'status_code'):
            status_code = error.status_code
        else:
            status_code = None
            
        if hasattr(error, 'response'):
            response_body = str(error.response)
        else:
            response_body = str(error)
        
        # Generic API error
        raise APIError(
            f"OpenAI API error: {str(error)}",
            status_code=status_code,
            response_body=response_body
        )


class OpenAITTSAdapter(OpenAIAdapter):
    """Specialized adapter for OpenAI TTS operations."""
    
    def __init__(self, api_key: str, default_voice: str = "alloy", default_model: str = "tts-1"):
        """
        Initialize TTS adapter.
        
        Args:
            api_key: OpenAI API key
            default_voice: Default voice to use
            default_model: Default model to use
        """
        super().__init__(api_key)
        self.default_voice = default_voice
        self.default_model = default_model
        
        # Available voices
        self.available_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        # Available models
        self.available_models = ["tts-1", "tts-1-hd"]
        
        logger.info(
            "OpenAI TTS adapter initialized",
            default_voice=default_voice,
            default_model=default_model
        )
    
    async def generate_speech_file(
        self,
        text: str,
        output_path: Path,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: str = "wav"
    ) -> Path:
        """
        Generate speech and save to file.
        
        Args:
            text: Text to convert
            output_path: Where to save the audio
            voice: Voice to use (defaults to instance default)
            model: Model to use (defaults to instance default)
            response_format: Audio format
            
        Returns:
            Path to generated audio file
        """
        voice = voice or self.default_voice
        model = model or self.default_model
        
        # Validate parameters
        if voice not in self.available_voices:
            raise APIError(f"Invalid voice: {voice}. Available: {self.available_voices}")
        
        if model not in self.available_models:
            raise APIError(f"Invalid model: {model}. Available: {self.available_models}")
        
        # Generate speech
        await self.text_to_speech(
            text=text,
            voice=voice,
            model=model,
            response_format=response_format,
            output_path=output_path
        )
        
        return output_path
    
    async def generate_speech_batch(
        self,
        texts: List[str],
        output_dir: Path,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: str = "wav",
        max_concurrent: int = 5
    ) -> List[Path]:
        """
        Generate speech for multiple texts with concurrency control.
        
        Args:
            texts: List of texts to convert
            output_dir: Directory to save audio files
            voice: Voice to use
            model: Model to use
            response_format: Audio format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of paths to generated audio files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(text: str, index: int) -> Path:
            async with semaphore:
                output_path = output_dir / f"speech_{index:04d}.{response_format}"
                return await self.generate_speech_file(
                    text=text,
                    output_path=output_path,
                    voice=voice,
                    model=model,
                    response_format=response_format
                )
        
        # Execute generation tasks
        tasks = [generate_single(text, i) for i, text in enumerate(texts)]
        audio_paths = await asyncio.gather(*tasks)
        
        logger.info(
            "Batch speech generation completed",
            text_count=len(texts),
            output_dir=str(output_dir)
        )
        
        return audio_paths