"""Transcript processing service implementation following ADR-0003."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..core.interfaces import TranscriptProcessingService
from ..core.models import (
    ProcessedSegment,
    ProcessingMode,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for transcript processing."""
    min_segment_duration: float = 0.5
    max_segment_duration: float = 30.0
    merge_threshold_seconds: float = 2.0
    min_words_per_segment: int = 2
    sentence_boundary_patterns: list[str] = None

    def __post_init__(self):
        if self.sentence_boundary_patterns is None:
            self.sentence_boundary_patterns = [
                r'[.!?]+\s*$',  # Sentence endings
                r'[.!?]+\s+[A-Z]',  # Sentence ending followed by capital
                r':\s*$',  # Colon endings
            ]


class HybridTranscriptProcessingService(TranscriptProcessingService):
    """
    Hybrid transcript processing service with rule-based and AI-powered modes.

    Implements the three-phase approach from ADR-0003:
    1. Rule-based preprocessing
    2. Optional AI enhancement
    3. Quality validation
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the transcript processing service.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()

    async def process_transcript(
        self,
        segments: list[TranscriptSegment],
        mode: ProcessingMode = ProcessingMode.HYBRID
    ) -> list[ProcessedSegment]:
        """
        Process transcript segments using the specified mode.

        Args:
            segments: Raw transcript segments to process
            mode: Processing mode (rule-based, AI-enhanced, or hybrid)

        Returns:
            Processed transcript segments ready for translation
        """
        logger.info(f"Processing {len(segments)} segments in {mode.value} mode")

        # Phase 1: Rule-based preprocessing
        preprocessed = await self._rule_based_preprocessing(segments)
        logger.debug(f"Rule-based preprocessing: {len(segments)} -> {len(preprocessed)} segments")

        # Phase 2: AI enhancement (if enabled)
        if mode in [ProcessingMode.AI_ENHANCED, ProcessingMode.HYBRID]:
            enhanced = await self._ai_enhance_segments(preprocessed, mode)
            logger.debug(f"AI enhancement completed for {len(enhanced)} segments")
            return enhanced

        return preprocessed

    async def validate_segments(self, segments: list[TranscriptSegment]) -> bool:
        """
        Validate transcript segments for quality and completeness.

        Args:
            segments: Transcript segments to validate

        Returns:
            True if segments are valid
        """
        if not segments:
            logger.warning("No segments to validate")
            return False

        for i, segment in enumerate(segments):
            # Check timing consistency
            if segment.start_time < 0 or segment.end_time <= segment.start_time:
                logger.error(f"Segment {i} has invalid timing: {segment.start_time}-{segment.end_time}")
                return False

            # Check for overlapping segments
            if i > 0 and segment.start_time < segments[i-1].end_time:
                logger.warning(f"Segment {i} overlaps with previous segment")

            # Check text content
            if not segment.text.strip():
                logger.error(f"Segment {i} has empty text")
                return False

        logger.info(f"Validated {len(segments)} segments successfully")
        return True

    async def _rule_based_preprocessing(self, segments: list[TranscriptSegment]) -> list[ProcessedSegment]:
        """
        Phase 1: Rule-based preprocessing.

        - Merge short segments
        - Fix basic punctuation
        - Detect sentence boundaries
        - Preserve timing information
        """
        if not segments:
            return []

        # Step 1: Filter and clean segments
        cleaned_segments = self._clean_segments(segments)

        # Step 2: Merge short adjacent segments
        merged_segments = self._merge_short_segments(cleaned_segments)

        # Step 3: Fix punctuation and capitalization
        processed_segments = []
        for sequence_num, (merged_group, original_indices) in enumerate(merged_segments):
            processed_text = self._fix_punctuation_and_capitalization(merged_group)

            # Create ProcessedSegment
            processed = ProcessedSegment(
                merged_segments=merged_group,
                processed_text=processed_text,
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=sequence_num,
                original_indices=original_indices,
                is_sentence_complete=self._is_sentence_complete(processed_text),
                context_quality_score=self._calculate_quality_score(processed_text, merged_group),
                timing_preserved=True,
                enhancement_metadata={
                    "rule_based_processing": True,
                    "segments_merged": len(merged_group),
                    "original_text": " ".join(s.text for s in merged_group),
                },
                ready_for_translation=True
            )
            processed_segments.append(processed)

        return processed_segments

    async def _ai_enhance_segments(
        self,
        segments: list[ProcessedSegment],
        mode: ProcessingMode
    ) -> list[ProcessedSegment]:
        """
        Phase 2: AI enhancement of segments.

        For now, this is a placeholder that preserves the rule-based processing.
        In a full implementation, this would call OpenAI API for context enhancement.
        """
        logger.info(f"AI enhancement mode: {mode.value}")

        enhanced_segments = []
        for segment in segments:
            # In hybrid mode, only enhance low-quality segments
            if mode == ProcessingMode.HYBRID and segment.context_quality_score and segment.context_quality_score > 0.7:
                logger.debug(f"Skipping AI enhancement for high-quality segment (score: {segment.context_quality_score})")
                enhanced_segments.append(segment)
                continue

            # Placeholder for AI enhancement
            # TODO: Implement OpenAI API call for context reconstruction
            enhanced_text = await self._placeholder_ai_enhancement(segment.processed_text)

            # Update segment with AI enhancement
            enhanced = ProcessedSegment(
                merged_segments=segment.merged_segments,
                processed_text=enhanced_text,
                processing_mode=mode,
                sequence_number=segment.sequence_number,
                original_indices=segment.original_indices,
                is_sentence_complete=self._is_sentence_complete(enhanced_text),
                context_quality_score=min(1.0, (segment.context_quality_score or 0.5) + 0.2),
                timing_preserved=segment.timing_preserved,
                enhancement_metadata={
                    **segment.enhancement_metadata,
                    "ai_enhanced": True,
                    "enhancement_applied": enhanced_text != segment.processed_text,
                },
                ready_for_translation=True
            )
            enhanced_segments.append(enhanced)

        return enhanced_segments

    def _clean_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Clean segments - normalize text but preserve all segments for merging."""
        cleaned = []
        for segment in segments:
            # Clean up text
            text = segment.text.strip()
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

            # Keep all segments with any meaningful text - filtering happens during merging
            if text:
                cleaned.append(TranscriptSegment(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    text=text
                ))

        return cleaned

    def _merge_short_segments(self, segments: list[TranscriptSegment]) -> list[tuple[list[TranscriptSegment], list[int]]]:
        """Merge short adjacent segments into groups, ensuring quality thresholds."""
        if not segments:
            return []

        merged_groups = []
        current_group = [segments[0]]
        current_indices = [0]

        for i, segment in enumerate(segments[1:], 1):
            prev_segment = current_group[-1]

            # Check if segments should be merged
            time_gap = segment.start_time - prev_segment.end_time
            current_duration = current_group[-1].end_time - current_group[0].start_time
            current_words = sum(len(s.text.split()) for s in current_group)

            # Force merge if current group doesn't meet quality thresholds
            force_merge = (
                current_duration < self.config.min_segment_duration or
                current_words < self.config.min_words_per_segment
            )

            should_merge = (
                force_merge or (
                    time_gap <= self.config.merge_threshold_seconds and
                    current_duration < self.config.max_segment_duration and
                    not self._is_sentence_complete(prev_segment.text)
                )
            )

            if should_merge:
                current_group.append(segment)
                current_indices.append(i)
            else:
                merged_groups.append((current_group, current_indices))
                current_group = [segment]
                current_indices = [i]

        # Add the last group
        if current_group:
            merged_groups.append((current_group, current_indices))

        # Final pass: merge any remaining groups that don't meet thresholds
        final_groups = []
        for group, indices in merged_groups:
            total_words = sum(len(s.text.split()) for s in group)
            total_duration = group[-1].end_time - group[0].start_time

            # If group still doesn't meet thresholds, try to merge with previous group
            if (total_words < self.config.min_words_per_segment or
                total_duration < self.config.min_segment_duration) and final_groups:
                # Merge with previous group
                final_groups[-1] = (
                    final_groups[-1][0] + group,
                    final_groups[-1][1] + indices
                )
            else:
                final_groups.append((group, indices))

        return final_groups

    def _fix_punctuation_and_capitalization(self, segments: list[TranscriptSegment]) -> str:
        """Fix punctuation and capitalization for a group of segments."""
        combined_text = " ".join(segment.text for segment in segments)

        # Basic punctuation fixes
        text = combined_text.strip()

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Add period if no ending punctuation
        if text and not re.search(r'[.!?]$', text):
            text += '.'

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

        return text

    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text represents a complete sentence."""
        text = text.strip()
        if not text:
            return False

        # Check for sentence ending punctuation - definitely complete
        if re.search(r'[.!?]$', text):
            return True

        # Check for common incomplete patterns - definitely incomplete
        incomplete_patterns = [
            r'\b(and|but|or|so|because|since|while|when|if|although)\s*$',
            r'\b(the|a|an)\s*$',
            r'\b(is|are|was|were|will|would|could|should|can|may|must)\s*$',
            r'\b(then|now|here|there)\s*$',
            r'\b(with|without|from|to|in|on|at|by|for)\s*$',
        ]

        for pattern in incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False

        # For short phrases without punctuation, be more conservative
        word_count = len(text.split())
        if word_count < 3:
            return False

        # If no clear incomplete markers and reasonable length, consider complete
        return True

    def _calculate_quality_score(self, text: str, segments: list[TranscriptSegment]) -> float:
        """Calculate a quality score for the processed text."""
        score = 0.5  # Base score

        # Sentence completeness
        if self._is_sentence_complete(text):
            score += 0.2

        # Text length (prefer medium-length segments)
        word_count = len(text.split())
        if 5 <= word_count <= 20:
            score += 0.1
        elif word_count > 30:
            score -= 0.1

        # Proper capitalization
        if text and text[0].isupper():
            score += 0.1

        # Proper punctuation
        if re.search(r'[.!?]$', text):
            score += 0.1

        # Timing consistency
        total_duration = segments[-1].end_time - segments[0].start_time
        if 1.0 <= total_duration <= 10.0:
            score += 0.1

        return min(1.0, max(0.0, score))

    async def _placeholder_ai_enhancement(self, text: str) -> str:
        """
        Placeholder for AI enhancement.

        In a full implementation, this would call OpenAI API to:
        - Complete fragmented sentences
        - Improve context flow
        - Optimize for translation quality
        """
        # For now, just return the original text
        # TODO: Implement OpenAI API integration
        return text
