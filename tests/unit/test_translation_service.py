"""Unit tests for TranslationService implementations."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from videodub.core.exceptions import TranslationError
from videodub.services.translator import (
    FallbackTranslationService,
    OpenAITranslationService,
)


class TestOpenAITranslationService:
    """Test OpenAI translation service."""

    @pytest.fixture
    def service(self):
        """Create translation service instance."""
        return OpenAITranslationService(api_key="test-key", model="gpt-4.1-nano")

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.content = "Hola, este es un texto de prueba."
        return response

    @pytest.mark.asyncio
    async def test_translate_text_success(self, service, mock_openai_response):
        """Test successful text translation."""
        with patch.object(
            service.client, "chat_completion", return_value=mock_openai_response
        ):
            result = await service.translate_text("Hello, this is a test text.", "es")
            assert result == "Hola, este es un texto de prueba."

    @pytest.mark.asyncio
    async def test_translate_text_empty_input(self, service):
        """Test translation with empty input."""
        result = await service.translate_text("", "es")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_text_whitespace_only(self, service):
        """Test translation with whitespace-only input."""
        result = await service.translate_text("   ", "es")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_text_invalid_language(self, service):
        """Test translation with invalid language code."""
        with pytest.raises(Exception):  # Should raise validation error
            await service.translate_text("Hello", "invalid-lang")

    @pytest.mark.asyncio
    async def test_translate_text_api_error(self, service):
        """Test translation with API error."""
        with patch.object(
            service.client, "chat_completion", side_effect=Exception("API Error")
        ):
            with pytest.raises(TranslationError):
                await service.translate_text("Hello", "es")

    @pytest.mark.asyncio
    async def test_translate_batch_success(self, service):
        """Test successful batch translation."""
        texts = ["Hello", "World", "Test"]
        expected_translations = ["Hola", "Mundo", "Prueba"]

        async def mock_translate_text(text, lang):
            mapping = {"Hello": "Hola", "World": "Mundo", "Test": "Prueba"}
            return mapping.get(text, f"[Translation of {text}]")

        with patch.object(service, "translate_text", side_effect=mock_translate_text):
            result = await service.translate_batch(texts, "es")
            assert result == expected_translations

    @pytest.mark.asyncio
    async def test_translate_batch_empty_list(self, service):
        """Test batch translation with empty list."""
        result = await service.translate_batch([], "es")
        assert result == []

    @pytest.mark.asyncio
    async def test_translate_batch_partial_failure(self, service):
        """Test batch translation with some failures."""
        texts = ["Hello", "World"]

        async def mock_translate_text(text, lang):
            if text == "Hello":
                return "Hola"
            else:
                raise Exception("Translation failed")

        with patch.object(service, "translate_text", side_effect=mock_translate_text):
            result = await service.translate_batch(texts, "es")
            assert result[0] == "Hola"
            assert result[1].startswith("[Translation failed:")

    @pytest.mark.asyncio
    async def test_translate_batch_concurrency(self, service):
        """Test batch translation processes concurrently."""
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        call_order = []

        async def mock_translate_text(text, lang):
            call_order.append(text)
            return f"Translated: {text}"

        with patch.object(service, "translate_text", side_effect=mock_translate_text):
            result = await service.translate_batch(texts, "es")

            # Check all texts were processed
            assert len(result) == 5
            assert all(r.startswith("Translated:") for r in result)

            # Check they were called (order may vary due to concurrency)
            assert len(call_order) == 5
            assert set(call_order) == set(texts)


class TestFallbackTranslationService:
    """Test fallback translation service."""

    @pytest.fixture
    def service(self):
        """Create fallback translation service instance."""
        return FallbackTranslationService()

    @pytest.mark.asyncio
    async def test_translate_text_with_prefix(self, service):
        """Test fallback translation adds language prefix."""
        result = await service.translate_text("Hello, world!", "es")
        assert result == "[ES] Hello, world!"

    @pytest.mark.asyncio
    async def test_translate_text_unknown_language(self, service):
        """Test fallback translation with unknown language."""
        result = await service.translate_text("Hello", "xx")  # Use valid 2-char code
        assert result == "[XX] Hello"

    @pytest.mark.asyncio
    async def test_translate_text_empty_input(self, service):
        """Test fallback translation with empty input."""
        result = await service.translate_text("", "es")
        assert result == ""

    @pytest.mark.asyncio
    async def test_translate_batch_success(self, service):
        """Test successful fallback batch translation."""
        texts = ["Hello", "World", "Test"]
        result = await service.translate_batch(texts, "es")

        expected = ["[ES] Hello", "[ES] World", "[ES] Test"]
        assert result == expected

    @pytest.mark.asyncio
    async def test_translate_batch_empty_list(self, service):
        """Test fallback batch translation with empty list."""
        result = await service.translate_batch([], "es")
        assert result == []

    @pytest.mark.asyncio
    async def test_translate_batch_mixed_languages(self, service):
        """Test fallback batch translation with different languages."""
        texts = ["Hello", "World"]

        # Test Spanish
        result_es = await service.translate_batch(texts, "es")
        assert result_es == ["[ES] Hello", "[ES] World"]

        # Test French
        result_fr = await service.translate_batch(texts, "fr")
        assert result_fr == ["[FR] Hello", "[FR] World"]

    @pytest.mark.asyncio
    async def test_all_supported_languages(self, service):
        """Test fallback translation with all supported languages."""
        text = "Hello"
        supported_languages = [
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ja",
            "ko",
            "zh",
            "ru",
            "ar",
            "hi",
        ]

        for lang in supported_languages:
            result = await service.translate_text(text, lang)
            assert result == f"[{lang.upper()}] {text}"


class TestTranslationServiceInterface:
    """Test that both services implement the interface correctly."""

    @pytest.mark.asyncio
    async def test_interface_compliance(self):
        """Test that both services have the required methods."""
        openai_service = OpenAITranslationService(api_key="test-key")
        fallback_service = FallbackTranslationService()

        # Check both services have required methods
        assert hasattr(openai_service, "translate_text")
        assert hasattr(openai_service, "translate_batch")
        assert hasattr(fallback_service, "translate_text")
        assert hasattr(fallback_service, "translate_batch")

        # Check methods are callable
        assert callable(openai_service.translate_text)
        assert callable(openai_service.translate_batch)
        assert callable(fallback_service.translate_text)
        assert callable(fallback_service.translate_batch)

    @pytest.mark.asyncio
    async def test_method_signatures(self):
        """Test that method signatures match interface."""
        import inspect

        openai_service = OpenAITranslationService(api_key="test-key")
        fallback_service = FallbackTranslationService()

        # Check translate_text signature
        openai_sig = inspect.signature(openai_service.translate_text)
        fallback_sig = inspect.signature(fallback_service.translate_text)

        assert len(openai_sig.parameters) == 2  # text, target_language
        assert len(fallback_sig.parameters) == 2  # text, target_language

        # Check translate_batch signature
        openai_batch_sig = inspect.signature(openai_service.translate_batch)
        fallback_batch_sig = inspect.signature(fallback_service.translate_batch)

        assert len(openai_batch_sig.parameters) == 2  # texts, target_language
        assert len(fallback_batch_sig.parameters) == 2  # texts, target_language
