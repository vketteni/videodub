"""Cost tracking utilities for OpenAI API usage."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OpenAIModel(Enum):
    """OpenAI model identifiers with pricing."""

    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_1_NANO = "gpt-4.1-nano"
    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"


@dataclass
class ModelPricing:
    """Pricing information for OpenAI models."""

    input_price_per_1k: float  # USD per 1K tokens
    output_price_per_1k: float  # USD per 1K tokens
    audio_price_per_1k_chars: Optional[float] = None  # USD per 1K characters for TTS


# Current OpenAI pricing (as of 2024 - update as needed)
MODEL_PRICING = {
    OpenAIModel.GPT_35_TURBO: ModelPricing(0.0005, 0.0015),
    OpenAIModel.GPT_4: ModelPricing(0.03, 0.06),
    OpenAIModel.GPT_4_TURBO: ModelPricing(0.01, 0.03),
    OpenAIModel.GPT_4_1_NANO: ModelPricing(
        0.0001, 0.0003
    ),  # Estimated - adjust when available
    OpenAIModel.TTS_1: ModelPricing(0, 0, 0.015),  # $0.015 per 1K characters
    OpenAIModel.TTS_1_HD: ModelPricing(0, 0, 0.030),  # $0.030 per 1K characters
}


@dataclass
class UsageStats:
    """Token/character usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    characters: int = 0  # For TTS
    requests: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if self.total_tokens == 0 and (self.input_tokens > 0 or self.output_tokens > 0):
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""

    model: str
    input_tokens: int
    output_tokens: int
    characters: int
    input_cost: float
    output_cost: float
    audio_cost: float
    total_cost: float
    requests: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "characters": self.characters,
                "requests": self.requests,
            },
            "costs": {
                "input_cost": round(self.input_cost, 6),
                "output_cost": round(self.output_cost, 6),
                "audio_cost": round(self.audio_cost, 6),
                "total_cost": round(self.total_cost, 6),
            },
        }


class CostTracker:
    """Tracks and calculates costs for OpenAI API usage."""

    def __init__(self):
        self.usage_by_model: Dict[str, UsageStats] = {}
        self.session_start = datetime.now()

    def track_chat_completion(self, model: str, usage_data: Dict) -> None:
        """
        Track usage from OpenAI chat completion response.

        Args:
            model: Model name (e.g., "gpt-4.1-nano")
            usage_data: Usage data from OpenAI response.usage
        """
        if model not in self.usage_by_model:
            self.usage_by_model[model] = UsageStats()

        stats = self.usage_by_model[model]
        stats.input_tokens += usage_data.get("prompt_tokens", 0)
        stats.output_tokens += usage_data.get("completion_tokens", 0)
        stats.total_tokens += usage_data.get("total_tokens", 0)
        stats.requests += 1

    def track_tts_usage(self, model: str, character_count: int) -> None:
        """
        Track TTS usage.

        Args:
            model: TTS model name (e.g., "tts-1")
            character_count: Number of characters processed
        """
        if model not in self.usage_by_model:
            self.usage_by_model[model] = UsageStats()

        stats = self.usage_by_model[model]
        stats.characters += character_count
        stats.requests += 1

    def calculate_cost(self, model_name: str) -> CostBreakdown:
        """
        Calculate cost for a specific model.

        Args:
            model_name: Name of the model to calculate cost for

        Returns:
            CostBreakdown with detailed cost information
        """
        if model_name not in self.usage_by_model:
            return CostBreakdown(
                model=model_name,
                input_tokens=0,
                output_tokens=0,
                characters=0,
                input_cost=0.0,
                output_cost=0.0,
                audio_cost=0.0,
                total_cost=0.0,
                requests=0,
            )

        stats = self.usage_by_model[model_name]

        # Find pricing for model
        pricing = None
        for model_enum in OpenAIModel:
            if model_enum.value == model_name:
                pricing = MODEL_PRICING.get(model_enum)
                break

        if not pricing:
            # Unknown model, use GPT-4 pricing as fallback
            pricing = MODEL_PRICING[OpenAIModel.GPT_4]

        # Calculate costs
        input_cost = (stats.input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (stats.output_tokens / 1000) * pricing.output_price_per_1k
        audio_cost = 0.0

        if pricing.audio_price_per_1k_chars and stats.characters > 0:
            audio_cost = (stats.characters / 1000) * pricing.audio_price_per_1k_chars

        total_cost = input_cost + output_cost + audio_cost

        return CostBreakdown(
            model=model_name,
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            characters=stats.characters,
            input_cost=input_cost,
            output_cost=output_cost,
            audio_cost=audio_cost,
            total_cost=total_cost,
            requests=stats.requests,
        )

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        total = 0.0
        for model_name in self.usage_by_model:
            breakdown = self.calculate_cost(model_name)
            total += breakdown.total_cost
        return total

    def get_session_summary(self) -> Dict:
        """Get complete session summary with costs and usage."""
        summary = {
            "session_start": self.session_start.isoformat(),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "models": {},
            "totals": {
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_characters": 0,
                "total_requests": 0,
            },
        }

        for model_name in self.usage_by_model:
            cost_breakdown = self.calculate_cost(model_name)
            summary["models"][model_name] = cost_breakdown.to_dict()

            # Update totals
            summary["totals"]["total_cost"] += cost_breakdown.total_cost
            summary["totals"]["total_tokens"] += (
                cost_breakdown.input_tokens + cost_breakdown.output_tokens
            )
            summary["totals"]["total_characters"] += cost_breakdown.characters
            summary["totals"]["total_requests"] += cost_breakdown.requests

        # Round total cost
        summary["totals"]["total_cost"] = round(summary["totals"]["total_cost"], 6)

        return summary

    def reset(self) -> None:
        """Reset all tracking data."""
        self.usage_by_model.clear()
        self.session_start = datetime.now()


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_global_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_cost_tracker() -> None:
    """Reset the global cost tracker."""
    global _global_tracker
    if _global_tracker:
        _global_tracker.reset()


def track_openai_usage(model: str, usage_data: Dict) -> None:
    """Convenience function to track OpenAI usage globally."""
    tracker = get_global_cost_tracker()
    tracker.track_chat_completion(model, usage_data)


def track_tts_usage(model: str, character_count: int) -> None:
    """Convenience function to track TTS usage globally."""
    tracker = get_global_cost_tracker()
    tracker.track_tts_usage(model, character_count)


def get_session_cost_summary() -> Dict:
    """Get current session cost summary."""
    tracker = get_global_cost_tracker()
    return tracker.get_session_summary()
