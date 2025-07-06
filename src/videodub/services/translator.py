"""Translation service implementation."""

import asyncio

import structlog

from ..adapters.openai_adapter import OpenAIAdapter
from ..config.validation import validate_language_code, validate_text_length
from ..core.exceptions import APIError, RateLimitError, TranslationError
from ..core.interfaces import TranslationService

logger = structlog.get_logger(__name__)


class OpenAITranslationService(TranslationService):
    """Translation service using OpenAI GPT models."""

    def __init__(self, api_key: str, model: str = "gpt-4.1-nano", max_retries: int = 3):
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

        logger.debug(
            "Translating text", target_language=target_language, text_length=len(text)
        )

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
                        {"role": "user", "content": text},
                    ],
                    max_tokens=len(text) * 2,  # Allow for expansion
                    temperature=0.3,  # Lower temperature for more consistent translations
                )

                translated_text = response.choices[0].message.content.strip()

                if not translated_text:
                    raise TranslationError("Empty translation received")

                logger.debug(
                    "Translation completed",
                    target_language=target_language,
                    original_length=len(text),
                    translated_length=len(translated_text),
                )

                return translated_text

            except RateLimitError:
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        "Rate limit hit, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise

            except APIError as e:
                if attempt < self.max_retries and e.status_code >= 500:
                    wait_time = 2**attempt
                    logger.warning(
                        "API error, retrying",
                        attempt=attempt + 1,
                        status_code=e.status_code,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise TranslationError(
                    f"API error: {e.message}", details={"status_code": e.status_code}
                )

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Translation attempt failed, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(1)
                    continue

                logger.error("Translation failed after all retries", error=str(e))
                raise TranslationError(f"Translation failed: {str(e)}")

    async def translate_batch(
        self, texts: list[str], target_language: str
    ) -> list[str]:
        """
        Translate multiple text strings.

        Args:
            texts: List of text strings to translate
            target_language: Target language code

        Returns:
            List of translated text strings

        Raises:
            TranslationError: If translation fails
        """
        validate_language_code(target_language)

        if not texts:
            return []

        logger.info(
            "Starting batch translation",
            target_language=target_language,
            text_count=len(texts),
        )

        # Process texts with concurrency control
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def translate_single_text(text: str) -> str:
            async with semaphore:
                try:
                    return await self.translate_text(text, target_language)
                except Exception as e:
                    logger.error(
                        "Failed to translate text",
                        text=text[:50],
                        error=str(e),
                    )
                    # Return fallback translation
                    return f"[Translation failed: {text}]"

        # Execute translations concurrently
        tasks = [translate_single_text(text) for text in texts]
        translated_texts = await asyncio.gather(*tasks, return_exceptions=False)

        successful_translations = len(
            [
                text
                for text in translated_texts
                if not text.startswith("[Translation failed:")
            ]
        )

        logger.info(
            "Batch translation completed",
            target_language=target_language,
            total_texts=len(texts),
            successful_translations=successful_translations,
            failed_translations=len(texts) - successful_translations,
        )

        return translated_texts


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
            "hi": "[HI] ",
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

        prefix = self.language_prefixes.get(
            target_language, f"[{target_language.upper()}] "
        )
        return f"{prefix}{text}"

    async def translate_batch(
        self, texts: list[str], target_language: str
    ) -> list[str]:
        """
        Apply language prefix to all text strings.

        Args:
            texts: List of text strings
            target_language: Target language code

        Returns:
            List of "translated" text strings with prefixes
        """
        validate_language_code(target_language)

        translated_texts = []

        for text in texts:
            translated_text = await self.translate_text(text, target_language)
            translated_texts.append(translated_text)

        return translated_texts
