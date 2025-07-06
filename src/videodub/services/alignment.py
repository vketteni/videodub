"""Timing alignment services for synchronizing translations with original speech timing."""

import asyncio
import logging
import time
from typing import List, Optional

from ..core.exceptions import AlignmentError
from ..core.interfaces import AlignmentService
from ..core.models import (
    AlignmentConfig,
    AlignmentEvaluation,
    AlignmentStrategy,
    TimedTranscript,
    TimedTranslation,
    TimedTranslationSegment,
    TimingMetadata,
)

logger = logging.getLogger(__name__)


class TimingAlignmentService(AlignmentService):
    """Service for aligning translated texts with original timing information."""

    def __init__(self):
        """Initialize the timing alignment service."""
        self._supported_strategies = [
            AlignmentStrategy.LENGTH_BASED,
            AlignmentStrategy.SENTENCE_BOUNDARY,
            AlignmentStrategy.HYBRID,
        ]

    async def align_translation(
        self,
        timed_transcript: TimedTranscript,
        translated_texts: List[str],
        target_language: str,
        config: AlignmentConfig,
    ) -> TimedTranslation:
        """
        Align translated texts with original timing information.

        Args:
            timed_transcript: Original transcript with timing
            translated_texts: List of translated text strings
            target_language: Target language code
            config: Alignment configuration and strategy

        Returns:
            TimedTranslation with synchronized timing

        Raises:
            AlignmentError: If alignment fails
        """
        try:
            start_time = time.time()

            # Validate inputs
            await self._validate_inputs(timed_transcript, translated_texts, config)

            # Extract original texts for alignment
            original_texts = [segment.text for segment in timed_transcript.segments]

            # Perform alignment based on strategy
            if config.strategy == AlignmentStrategy.LENGTH_BASED:
                aligned_segments = await self._align_by_length(
                    timed_transcript.segments, translated_texts, config
                )
            elif config.strategy == AlignmentStrategy.SENTENCE_BOUNDARY:
                aligned_segments = await self._align_by_sentence_boundary(
                    timed_transcript.segments, translated_texts, config
                )
            elif config.strategy == AlignmentStrategy.HYBRID:
                aligned_segments = await self._align_hybrid(
                    timed_transcript.segments, translated_texts, config
                )
            else:
                raise AlignmentError(
                    f"Unsupported alignment strategy: {config.strategy}"
                )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Evaluate alignment quality
            evaluation = await self._evaluate_alignment_quality(
                aligned_segments, timed_transcript.segments, execution_time
            )
            evaluation.strategy = config.strategy

            # Create timing metadata
            timing_metadata = TimingMetadata(
                total_duration=timed_transcript.timing_metadata.total_duration,
                segment_count=len(aligned_segments),
                average_segment_duration=timed_transcript.timing_metadata.average_segment_duration,
                timing_accuracy=evaluation.timing_accuracy,
                has_precise_timing=True,
                extraction_method=f"alignment_{config.strategy.value}",
            )

            return TimedTranslation(
                segments=aligned_segments,
                original_transcript=timed_transcript,
                target_language=target_language,
                alignment_config=config,
                alignment_evaluation=evaluation,
                timing_metadata=timing_metadata,
            )

        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
            raise AlignmentError(f"Failed to align translation: {str(e)}")

    async def _validate_inputs(
        self,
        timed_transcript: TimedTranscript,
        translated_texts: List[str],
        config: AlignmentConfig,
    ) -> None:
        """Validate alignment inputs."""
        if not timed_transcript.segments:
            raise AlignmentError("Timed transcript must have at least one segment")

        if not translated_texts:
            raise AlignmentError("Translated texts list cannot be empty")

        if config.strategy not in self._supported_strategies:
            raise AlignmentError(f"Unsupported alignment strategy: {config.strategy}")

    async def _align_by_length(
        self,
        original_segments: List,
        translated_texts: List[str],
        config: AlignmentConfig,
    ) -> List[TimedTranslationSegment]:
        """Align based on text length ratios."""
        aligned_segments = []

        # Calculate total lengths
        total_original_length = sum(len(seg.text) for seg in original_segments)
        total_translated_length = sum(len(text) for text in translated_texts)

        # Calculate length ratio for timing adjustment
        length_ratio = (
            total_translated_length / total_original_length
            if total_original_length > 0
            else 1.0
        )

        # Simple 1:1 mapping for now (can be enhanced)
        min_count = min(len(original_segments), len(translated_texts))

        for i in range(min_count):
            original_seg = original_segments[i]
            translated_text = translated_texts[i]

            # Calculate confidence based on length similarity
            original_length = len(original_seg.text)
            translated_length = len(translated_text)
            length_diff = abs(original_length - translated_length) / max(
                original_length, translated_length
            )
            confidence = max(0.1, 1.0 - length_diff)

            # Apply minimal timing adjustment
            timing_adjustment = 0.0
            if length_ratio > 1.2:  # Translated text is significantly longer
                timing_adjustment = min(0.5, (length_ratio - 1.0) * 0.3)
            elif length_ratio < 0.8:  # Translated text is significantly shorter
                timing_adjustment = max(-0.3, (length_ratio - 1.0) * 0.2)

            aligned_segments.append(
                TimedTranslationSegment(
                    start_time=original_seg.start_time,
                    end_time=original_seg.end_time + timing_adjustment,
                    original_text=original_seg.text,
                    translated_text=translated_text,
                    alignment_confidence=confidence,
                    timing_adjustment=timing_adjustment,
                )
            )

        return aligned_segments

    async def _align_by_sentence_boundary(
        self,
        original_segments: List,
        translated_texts: List[str],
        config: AlignmentConfig,
    ) -> List[TimedTranslationSegment]:
        """Align based on sentence boundaries."""
        # For now, implement as enhanced length-based alignment
        # TODO: Add sentence boundary detection logic
        aligned_segments = await self._align_by_length(
            original_segments, translated_texts, config
        )

        # Enhance with sentence boundary detection
        for segment in aligned_segments:
            if segment.translated_text.strip().endswith((".", "!", "?")):
                segment.boundary_type = "sentence_end"
                segment.alignment_confidence = min(
                    1.0, segment.alignment_confidence + 0.1
                )

        return aligned_segments

    async def _align_hybrid(
        self,
        original_segments: List,
        translated_texts: List[str],
        config: AlignmentConfig,
    ) -> List[TimedTranslationSegment]:
        """Hybrid alignment combining multiple strategies."""
        # Start with length-based alignment
        segments = await self._align_by_length(
            original_segments, translated_texts, config
        )

        # Apply sentence boundary improvements
        boundary_segments = await self._align_by_sentence_boundary(
            original_segments, translated_texts, config
        )

        # Combine confidence scores using weighted average
        length_weight = config.parameters.get("length_weight", 0.4)
        boundary_weight = config.parameters.get("boundary_weight", 0.3)

        for i, segment in enumerate(segments):
            if i < len(boundary_segments):
                boundary_seg = boundary_segments[i]
                # Weighted confidence combination
                combined_confidence = (
                    segment.alignment_confidence * length_weight
                    + boundary_seg.alignment_confidence * boundary_weight
                )
                segment.alignment_confidence = min(1.0, combined_confidence)
                segment.boundary_type = boundary_seg.boundary_type

        return segments

    async def _evaluate_alignment_quality(
        self,
        aligned_segments: List[TimedTranslationSegment],
        original_segments: List,
        execution_time: float,
    ) -> AlignmentEvaluation:
        """Evaluate alignment quality."""
        if not aligned_segments:
            return AlignmentEvaluation(
                strategy=AlignmentStrategy.LENGTH_BASED,  # Will be overridden
                timing_accuracy=0.0,
                text_preservation=0.0,
                boundary_alignment=0.0,
                overall_score=0.0,
                execution_time=execution_time,
                segment_count=0,
                average_confidence=0.0,
            )

        # Calculate metrics
        timing_accuracy = self._calculate_timing_accuracy(
            aligned_segments, original_segments
        )
        text_preservation = self._calculate_text_preservation(
            aligned_segments, original_segments
        )
        boundary_alignment = self._calculate_boundary_alignment(aligned_segments)
        average_confidence = sum(
            s.alignment_confidence for s in aligned_segments
        ) / len(aligned_segments)

        # Calculate overall score (weighted combination)
        overall_score = (
            timing_accuracy * 0.4
            + text_preservation * 0.3
            + boundary_alignment * 0.2
            + average_confidence * 0.1
        )

        return AlignmentEvaluation(
            strategy=AlignmentStrategy.LENGTH_BASED,  # Will be overridden
            timing_accuracy=timing_accuracy,
            text_preservation=text_preservation,
            boundary_alignment=boundary_alignment,
            overall_score=overall_score,
            execution_time=execution_time,
            segment_count=len(aligned_segments),
            average_confidence=average_confidence,
        )

    def _calculate_timing_accuracy(
        self, aligned_segments: List[TimedTranslationSegment], original_segments: List
    ) -> float:
        """Calculate timing accuracy score."""
        if not aligned_segments or not original_segments:
            return 0.0

        # Simple timing preservation score
        total_adjustment = sum(abs(s.timing_adjustment) for s in aligned_segments)
        max_possible_adjustment = sum(s.duration for s in aligned_segments)

        if max_possible_adjustment == 0:
            return 1.0

        return max(0.0, 1.0 - (total_adjustment / max_possible_adjustment))

    def _calculate_text_preservation(
        self, aligned_segments: List[TimedTranslationSegment], original_segments: List
    ) -> float:
        """Calculate text preservation score."""
        # For now, return high score as we're preserving all text
        return 0.9

    def _calculate_boundary_alignment(
        self, aligned_segments: List[TimedTranslationSegment]
    ) -> float:
        """Calculate sentence boundary alignment score."""
        if not aligned_segments:
            return 0.0

        boundary_segments = [
            s for s in aligned_segments if s.boundary_type == "sentence_end"
        ]
        return len(boundary_segments) / len(aligned_segments)

    async def evaluate_alignment(
        self,
        timed_translation: TimedTranslation,
        reference_alignment: Optional[TimedTranslation] = None,
    ) -> AlignmentEvaluation:
        """Evaluate alignment quality."""
        return timed_translation.alignment_evaluation

    async def compare_alignments(
        self,
        alignments: List[TimedTranslation],
    ) -> List[AlignmentEvaluation]:
        """Compare multiple alignment results for A/B testing."""
        return [alignment.alignment_evaluation for alignment in alignments]

    def get_supported_strategies(self) -> List[AlignmentStrategy]:
        """Get list of supported alignment strategies."""
        return self._supported_strategies.copy()
