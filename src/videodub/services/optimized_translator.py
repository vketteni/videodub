"""Optimized translation service with batch processing improvements."""

import asyncio
from typing import List

import structlog

from ..adapters.openai_adapter import OpenAIAdapter
from ..config.validation import validate_language_code
from ..core.exceptions import TranslationError
from ..core.interfaces import TranslationService

logger = structlog.get_logger(__name__)


class OptimizedOpenAITranslationService(TranslationService):
    """Enhanced translation service with batch optimization."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-nano",
        max_retries: int = 3,
        batch_size: int = 10,
        max_concurrent_batches: int = 3,
    ):
        """
        Initialize the optimized translation service.

        Args:
            api_key: OpenAI API key
            model: GPT model to use for translation
            max_retries: Maximum number of retry attempts
            batch_size: Number of texts to process in a single API call
            max_concurrent_batches: Number of batches to process concurrently
        """
        self.client = OpenAIAdapter(api_key)
        self.model = model
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.max_text_length = 4000

        logger.info(
            "Optimized translation service initialized",
            model=model,
            batch_size=batch_size,
            max_concurrent_batches=max_concurrent_batches,
        )

    async def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate a single text string.

        Args:
            text: Text to translate
            target_language: Target language code

        Returns:
            Translated text
        """
        # For single text, use batch method for consistency
        results = await self.translate_batch([text], target_language)
        return results[0]

    async def translate_batch(
        self, texts: List[str], target_language: str
    ) -> List[str]:
        """
        Translate multiple text strings using optimized batching.

        Args:
            texts: List of text strings to translate
            target_language: Target language code

        Returns:
            List of translated text strings
        """
        validate_language_code(target_language)

        if not texts:
            return []

        logger.info(
            "Starting optimized batch translation",
            target_language=target_language,
            text_count=len(texts),
            batch_size=self.batch_size,
        )

        # Split texts into batches
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        logger.debug(
            "Created translation batches",
            total_batches=len(batches),
            batch_size=self.batch_size,
        )

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_batch(batch: List[str], batch_index: int) -> List[str]:
            async with semaphore:
                return await self._translate_batch_optimized(
                    batch, target_language, batch_index
                )

        # Execute batch translations concurrently
        tasks = [
            process_batch(batch, i) for i, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        translated_texts = []
        for batch_result in batch_results:
            translated_texts.extend(batch_result)

        successful_translations = len([
            text for text in translated_texts 
            if not text.startswith("[Translation failed:")
        ])

        logger.info(
            "Optimized batch translation completed",
            target_language=target_language,
            total_texts=len(texts),
            successful_translations=successful_translations,
            failed_translations=len(texts) - successful_translations,
            batches_processed=len(batches),
        )

        return translated_texts

    async def _translate_batch_optimized(
        self, texts: List[str], target_language: str, batch_index: int
    ) -> List[str]:
        """
        Translate a batch of texts using a single API call.

        Args:
            texts: List of text strings to translate
            target_language: Target language code
            batch_index: Index of the batch for logging

        Returns:
            List of translated text strings
        """
        if not texts:
            return []

        # Language mapping for better prompts
        language_names = {
            "es": "Spanish",
            "fr": "French", 
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
        }

        target_lang_name = language_names.get(target_language, target_language.upper())

        # Create batch prompt with numbered texts
        batch_prompt = "Translate the following texts to " + target_lang_name + ". "
        batch_prompt += "Return ONLY the translations, one per line, in the same order:\\n\\n"
        
        for i, text in enumerate(texts, 1):
            batch_prompt += f"{i}. {text}\\n"

        system_prompt = (
            f"You are a professional translator. Translate each numbered text to {target_lang_name}. "
            "Maintain the original meaning, tone, and style. Return only the translated texts, "
            "one per line, in the same order as the input. Do not include numbers or other formatting."
        )

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    "Processing translation batch",
                    batch_index=batch_index,
                    batch_size=len(texts),
                    attempt=attempt + 1,
                )

                response = await self.client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": batch_prompt},
                    ],
                    max_tokens=sum(len(text) for text in texts) * 3,  # Allow for expansion
                    temperature=0.3,
                )

                translated_content = response.choices[0].message.content.strip()
                
                if not translated_content:
                    raise TranslationError("Empty translation received")

                # Split response into individual translations
                translated_lines = [
                    line.strip() 
                    for line in translated_content.split('\\n') 
                    if line.strip()
                ]

                # Ensure we have the same number of translations as inputs
                if len(translated_lines) != len(texts):
                    logger.warning(
                        "Translation count mismatch",
                        expected=len(texts),
                        received=len(translated_lines),
                        batch_index=batch_index,
                    )
                    
                    # Pad or trim to match expected count
                    if len(translated_lines) < len(texts):
                        # Pad with fallback translations
                        for i in range(len(translated_lines), len(texts)):
                            translated_lines.append(f"[Translation incomplete: {texts[i]}]")
                    else:
                        # Trim excess translations
                        translated_lines = translated_lines[:len(texts)]

                logger.debug(
                    "Batch translation completed",
                    batch_index=batch_index,
                    original_count=len(texts),
                    translated_count=len(translated_lines),
                )

                return translated_lines

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Batch translation attempt failed, retrying",
                        batch_index=batch_index,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                    continue

                logger.error(
                    "Batch translation failed after all retries",
                    batch_index=batch_index,
                    error=str(e),
                )
                
                # Return fallback translations for all texts in the batch
                return [f"[Translation failed: {text}]" for text in texts]

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return ["es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ru", "ar", "hi"]


class ConcurrentTTSOptimizer:
    """Helper class for optimizing TTS generation with concurrency."""
    
    def __init__(self, max_concurrent_files: int = 5):
        """
        Initialize TTS optimizer.
        
        Args:
            max_concurrent_files: Maximum number of audio files to generate concurrently
        """
        self.max_concurrent_files = max_concurrent_files
        self.semaphore = asyncio.Semaphore(max_concurrent_files)
        
        logger.info(
            "TTS optimizer initialized",
            max_concurrent_files=max_concurrent_files,
        )
    
    async def optimize_audio_generation(self, audio_generation_func, jobs):
        """
        Optimize audio generation with controlled concurrency.
        
        Args:
            audio_generation_func: Async function that generates audio
            jobs: List of audio generation jobs
            
        Returns:
            List of generated audio file paths
        """
        async def generate_with_semaphore(job):
            async with self.semaphore:
                return await audio_generation_func(job)
        
        logger.info(
            "Starting optimized audio generation",
            total_jobs=len(jobs),
            max_concurrent=self.max_concurrent_files,
        )
        
        tasks = [generate_with_semaphore(job) for job in jobs]
        results = await asyncio.gather(*tasks)
        
        logger.info(
            "Optimized audio generation completed",
            generated_files=len(results),
        )
        
        return results