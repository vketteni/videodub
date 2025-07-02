"""Translation service implementation."""

import asyncio
from typing import List, AsyncIterator, Optional

import structlog

from ..core.interfaces import TranslationService
from ..core.models import TranscriptSegment, TranslationSegment, TranslationJob
from ..core.exceptions import TranslationError, APIError, RateLimitError
from ..config.validation import validate_language_code, validate_text_length
from ..adapters.openai_adapter import OpenAIAdapter
from .transcript_processor import SentenceSegment, TranscriptProcessor


logger = structlog.get_logger(__name__)


class OpenAITranslationService(TranslationService):
    """Translation service using OpenAI GPT models."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        Initialize the translation service.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use for translation
            max_retries: Maximum number of retry attempts
        """
        self.client = OpenAIAdapter(api_key)
        self.model = model
        self.max_retries = max_retries
        self.max_text_length = 4000  # Conservative limit for GPT
        
        logger.info("Translation service initialized", model=model)
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate a single text string.
        
        Args:
            text: Text to translate
            target_language: Target language code
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        validate_language_code(target_language)
        validate_text_length(text, self.max_text_length, "input text")
        
        if not text.strip():
            return ""
        
        logger.debug("Translating text", target_language=target_language, text_length=len(text))
        
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
            "hi": "Hindi"
        }
        
        target_lang_name = language_names.get(target_language, target_language.upper())
        
        system_prompt = (
            f"You are a professional translator. Translate the following text to {target_lang_name}. "
            "Maintain the original meaning, tone, and style. Only return the translated text, nothing else."
        )
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=len(text) * 2,  # Allow for expansion
                    temperature=0.3  # Lower temperature for more consistent translations
                )
                
                translated_text = response.choices[0].message.content.strip()
                
                if not translated_text:
                    raise TranslationError("Empty translation received")
                
                logger.debug(
                    "Translation completed",
                    target_language=target_language,
                    original_length=len(text),
                    translated_length=len(translated_text)
                )
                
                return translated_text
                
            except RateLimitError:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        "Rate limit hit, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise
                
            except APIError as e:
                if attempt < self.max_retries and e.status_code >= 500:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "API error, retrying",
                        attempt=attempt + 1,
                        status_code=e.status_code,
                        wait_time=wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise TranslationError(f"API error: {e.message}", details={"status_code": e.status_code})
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Translation attempt failed, retrying",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    await asyncio.sleep(1)
                    continue
                
                logger.error("Translation failed after all retries", error=str(e))
                raise TranslationError(f"Translation failed: {str(e)}")
    
    async def translate_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_language: str
    ) -> List[TranslationSegment]:
        """
        Translate multiple transcript segments.
        
        Args:
            segments: List of transcript segments to translate
            target_language: Target language code
            
        Returns:
            List of translated segments
            
        Raises:
            TranslationError: If translation fails
        """
        validate_language_code(target_language)
        
        if not segments:
            return []
        
        logger.info(
            "Starting batch translation",
            target_language=target_language,
            segment_count=len(segments)
        )
        
        translated_segments = []
        
        # Process segments with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def translate_segment(segment: TranscriptSegment) -> TranslationSegment:
            async with semaphore:
                try:
                    translated_text = await self.translate_text(segment.text, target_language)
                    return TranslationSegment(
                        original_segment=segment,
                        translated_text=translated_text,
                        language=target_language
                    )
                except Exception as e:
                    logger.error(
                        "Failed to translate segment",
                        segment_start=segment.start_time,
                        segment_text=segment.text[:50],
                        error=str(e)
                    )
                    # Return fallback translation
                    return TranslationSegment(
                        original_segment=segment,
                        translated_text=f"[Translation failed: {segment.text}]",
                        language=target_language
                    )
        
        # Execute translations concurrently
        tasks = [translate_segment(segment) for segment in segments]
        translated_segments = await asyncio.gather(*tasks, return_exceptions=False)
        
        successful_translations = len([
            seg for seg in translated_segments 
            if not seg.translated_text.startswith("[Translation failed:")
        ])
        
        logger.info(
            "Batch translation completed",
            target_language=target_language,
            total_segments=len(segments),
            successful_translations=successful_translations,
            failed_translations=len(segments) - successful_translations
        )
        
        return translated_segments
    
    async def translate_batch(
        self, 
        job: TranslationJob
    ) -> AsyncIterator[TranslationSegment]:
        """
        Translate segments in batches with progress tracking.
        
        Args:
            job: Translation job with segments to process
            
        Yields:
            Translated segments as they are completed
            
        Raises:
            TranslationError: If translation fails
        """
        validate_language_code(job.target_language)
        
        logger.info(
            "Starting translation job",
            video_id=job.video_id,
            target_language=job.target_language,
            total_segments=job.total_segments
        )
        
        # Process in smaller batches to provide progress updates
        batch_size = 10
        
        for i in range(0, len(job.segments), batch_size):
            batch_segments = job.segments[i:i + batch_size]
            
            logger.debug(
                "Processing translation batch",
                batch_start=i,
                batch_size=len(batch_segments),
                total_segments=job.total_segments
            )
            
            # Translate batch
            translated_batch = await self.translate_segments(batch_segments, job.target_language)
            
            # Yield translated segments
            for translated_segment in translated_batch:
                job.add_translated_segment(translated_segment)
                yield translated_segment
        
        logger.info(
            "Translation job completed",
            video_id=job.video_id,
            target_language=job.target_language,
            completed_segments=job.completed_segments,
            progress=job.progress_percentage
        )
    
class FallbackTranslationService(TranslationService):
    """Fallback translation service that adds language prefixes."""
    
    def __init__(self):
        """Initialize the fallback service."""
        self.language_prefixes = {
            "es": "[ES] ",
            "fr": "[FR] ",
            "de": "[DE] ",
            "it": "[IT] ",
            "pt": "[PT] ",
            "ja": "[JA] ",
            "ko": "[KO] ",
            "zh": "[ZH] ",
            "ru": "[RU] ",
            "ar": "[AR] ",
            "hi": "[HI] "
        }
        
        logger.info("Fallback translation service initialized")
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """
        Apply language prefix to text as fallback translation.
        
        Args:
            text: Text to "translate"
            target_language: Target language code
            
        Returns:
            Text with language prefix
        """
        validate_language_code(target_language)
        
        if not text.strip():
            return ""
        
        prefix = self.language_prefixes.get(target_language, f"[{target_language.upper()}] ")
        return f"{prefix}{text}"
    
    async def translate_segments(
        self, 
        segments: List[TranscriptSegment], 
        target_language: str
    ) -> List[TranslationSegment]:
        """
        Apply language prefix to all segments.
        
        Args:
            segments: List of transcript segments
            target_language: Target language code
            
        Returns:
            List of "translated" segments with prefixes
        """
        validate_language_code(target_language)
        
        translated_segments = []
        
        for segment in segments:
            translated_text = await self.translate_text(segment.text, target_language)
            translated_segments.append(TranslationSegment(
                original_segment=segment,
                translated_text=translated_text,
                language=target_language
            ))
        
        return translated_segments
    
    async def translate_batch(
        self, 
        job: TranslationJob
    ) -> AsyncIterator[TranslationSegment]:
        """
        Process translation job with fallback method.
        
        Args:
            job: Translation job
            
        Yields:
            Translated segments
        """
        for segment in job.segments:
            translated_text = await self.translate_text(segment.text, job.target_language)
            translated_segment = TranslationSegment(
                original_segment=segment,
                translated_text=translated_text,
                language=job.target_language
            )
            job.add_translated_segment(translated_segment)
            yield translated_segment