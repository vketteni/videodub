"""Transcript processing service implementation following ADR-0003."""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..core.interfaces import TranscriptEnhancementService, TranscriptProcessingService
from ..core.models import ProcessedSegment, ProcessingMode, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for transcript processing."""

    min_segment_duration: float = 0.5
    max_segment_duration: float = 30.0
    merge_threshold_seconds: float = 2.0
    min_words_per_segment: int = 2
    sentence_boundary_patterns: list[str] = None
    # AI enhancement configuration
    ai_enhancement_threshold: float = 0.8

    def __post_init__(self):
        if self.sentence_boundary_patterns is None:
            self.sentence_boundary_patterns = [
                r"[.!?]+\s*$",  # Sentence endings
                r"[.!?]+\s+[A-Z]",  # Sentence ending followed by capital
                r":\s*$",  # Colon endings
            ]


class HybridTranscriptProcessingService(TranscriptProcessingService):
    """
    Hybrid transcript processing service with rule-based and AI-powered modes.

    Implements the three-phase approach from ADR-0003:
    1. Rule-based preprocessing
    2. Optional AI enhancement
    3. Quality validation
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        enhancement_service: Optional[TranscriptEnhancementService] = None,
    ):
        """
        Initialize the transcript processing service.

        Args:
            config: Processing configuration
            enhancement_service: Optional AI enhancement service for hybrid/AI modes
        """
        self.config = config or ProcessingConfig()
        self.enhancement_service = enhancement_service

    async def process_transcript(
        self,
        segments: list[TranscriptSegment],
        mode: ProcessingMode = ProcessingMode.HYBRID,
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
        logger.debug(
            f"Rule-based preprocessing: {len(segments)} -> {len(preprocessed)} segments"
        )

        # Phase 2: AI enhancement (if enabled)
        if mode in [ProcessingMode.AI_ENHANCED, ProcessingMode.HYBRID]:
            if self.enhancement_service:
                enhanced = await self._ai_enhance_segments(preprocessed, mode)
                logger.debug(f"AI enhancement completed for {len(enhanced)} segments")
                return enhanced
            else:
                logger.warning(
                    "AI enhancement requested but no enhancement service provided"
                )

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
                logger.error(
                    f"Segment {i} has invalid timing: {segment.start_time}-{segment.end_time}"
                )
                return False

            # Check for overlapping segments
            if i > 0 and segment.start_time < segments[i - 1].end_time:
                logger.warning(f"Segment {i} overlaps with previous segment")

            # Check text content
            if not segment.text.strip():
                logger.error(f"Segment {i} has empty text")
                return False

        logger.info(f"Validated {len(segments)} segments successfully")
        return True

    async def _rule_based_preprocessing(
        self, segments: list[TranscriptSegment]
    ) -> list[ProcessedSegment]:
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
        for sequence_num, (merged_group, original_indices) in enumerate(
            merged_segments
        ):
            processed_text = self._fix_punctuation_and_capitalization(merged_group)

            # Create ProcessedSegment
            processed = ProcessedSegment(
                merged_segments=merged_group,
                processed_text=processed_text,
                processing_mode=ProcessingMode.RULE_BASED,
                sequence_number=sequence_num,
                original_indices=original_indices,
                is_sentence_complete=self._is_sentence_complete(processed_text)[0],
                context_quality_score=self._calculate_quality_score(
                    processed_text, merged_group
                ),
                timing_preserved=True,
                enhancement_metadata={
                    "rule_based_processing": True,
                    "segments_merged": len(merged_group),
                    "original_text": " ".join(s.text for s in merged_group),
                },
                ready_for_translation=True,
            )
            processed_segments.append(processed)

        return processed_segments

    async def _ai_enhance_segments(
        self, segments: list[ProcessedSegment], mode: ProcessingMode
    ) -> list[ProcessedSegment]:
        """
        Phase 2: AI enhancement of segments using the enhancement service.
        """
        logger.info(f"AI enhancement mode: {mode.value}")

        if mode == ProcessingMode.HYBRID:
            # In hybrid mode, use batch enhancement with quality threshold
            threshold = getattr(self.config, "ai_enhancement_threshold", 0.8)
            enhanced_segments = await self.enhancement_service.enhance_segments_batch(
                segments, threshold=threshold
            )
        else:
            # In AI_ENHANCED mode, enhance all segments individually
            enhanced_segments = []
            for i, segment in enumerate(segments):
                prev_segment = segments[i - 1] if i > 0 else None
                next_segment = segments[i + 1] if i < len(segments) - 1 else None

                enhanced_segment = await self.enhancement_service.enhance_segment(
                    segment, prev_segment, next_segment
                )
                enhanced_segments.append(enhanced_segment)

        return enhanced_segments

    def _clean_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[TranscriptSegment]:
        """Clean segments - normalize text but preserve all segments for merging."""
        cleaned = []
        for segment in segments:
            # Clean up text
            text = segment.text.strip()
            text = re.sub(r"\s+", " ", text)  # Normalize whitespace

            # Keep all segments with any meaningful text - filtering happens during merging
            if text:
                cleaned.append(
                    TranscriptSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=text,
                    )
                )

        return cleaned

    def _merge_short_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[tuple[list[TranscriptSegment], list[int]]]:
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
                current_duration < self.config.min_segment_duration
                or current_words < self.config.min_words_per_segment
            )

            should_merge = force_merge or (
                time_gap <= self.config.merge_threshold_seconds
                and current_duration < self.config.max_segment_duration
                and not self._is_sentence_complete(prev_segment.text)[0]
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
            if (
                total_words < self.config.min_words_per_segment
                or total_duration < self.config.min_segment_duration
            ) and final_groups:
                # Merge with previous group
                final_groups[-1] = (
                    final_groups[-1][0] + group,
                    final_groups[-1][1] + indices,
                )
            else:
                final_groups.append((group, indices))

        return final_groups

    def _fix_punctuation_and_capitalization(
        self, segments: list[TranscriptSegment]
    ) -> str:
        """Fix punctuation and capitalization for a group of segments."""
        combined_text = " ".join(segment.text for segment in segments)

        # Basic punctuation fixes
        text = combined_text.strip()

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Add period if no ending punctuation
        if text and not re.search(r"[.!?]$", text):
            text += "."

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.!?])", r"\1", text)
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        return text

    def _is_sentence_complete(self, text: str) -> tuple[bool, float]:
        """
        Check if text represents a complete sentence.

        Returns:
            tuple: (is_complete, confidence_score)
                - is_complete: Boolean indicating if sentence appears complete
                - confidence_score: Float 0.0-1.0 indicating confidence in completeness
        """
        text = text.strip()
        if not text:
            return False, 0.0

        # Check for sentence ending punctuation - high confidence complete
        if re.search(r"[.!?]$", text):
            return True, 0.9

        # Check for definite incomplete patterns - high confidence incomplete
        definite_incomplete_patterns = [
            r"\b(and|but|or|so|because|since|while|when|if|although)\s*$",
            r"\b(the|a|an)\s*$",
            r"\b(is|are|was|were|will|would|could|should|can|may|must)\s*$",
            r"\b(with|without|from|to|in|on|at|by|for)\s*$",
            r"\b(of|about|over|under|through|during|before|after)\s*$",
            r"\b(don\'t|doesn\'t|won\'t|can\'t|wouldn\'t|couldn\'t|shouldn\'t)\s*$",
            r"\b(that|which|who|whom|whose|where|when|why|how)\s*$",
        ]

        for pattern in definite_incomplete_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, 0.1

        # Check for questionable incomplete patterns - medium confidence incomplete
        questionable_patterns = [
            r"\b(then|now|here|there)\s*$",
            r"\b(wanting|trying|going|coming|looking|talking|thinking|saying)\s*$",
            r"\b(you|me|him|her|them|us|it)\s*$",  # Pronouns
            r"\b(wrong|right|good|bad|important|necessary|possible|difficult)\s*$",  # Adjectives
            r"\b(displacing|overlook|overlooking)\s*$",  # Participles
        ]

        for pattern in questionable_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, 0.3

        # For short phrases without punctuation, be conservative
        word_count = len(text.split())
        if word_count < 3:
            return False, 0.2
        elif word_count < 5:
            return True, 0.6  # Short but potentially complete

        # If no clear incomplete markers and reasonable length, likely complete
        return True, 0.8

    def _calculate_quality_score(
        self, text: str, segments: list[TranscriptSegment]
    ) -> float:
        """
        Calculate a quality score for the processed text.

        Score components:
        - Base score: 0.3
        - Sentence completeness: 0.0-0.4 (weighted by confidence)
        - Text length optimization: 0.0-0.15
        - Proper capitalization: 0.05
        - Proper punctuation: 0.05
        - Timing consistency: 0.0-0.05
        """
        score = 0.3  # Lower base score to allow for more discrimination

        # Sentence completeness with confidence weighting
        is_complete, confidence = self._is_sentence_complete(text)
        if is_complete:
            # Scale completeness score by confidence (0.1 to 0.4)
            completeness_score = 0.1 + (confidence * 0.3)
            score += completeness_score
        else:
            # Penalty for incomplete sentences, scaled by confidence
            # High confidence incomplete (0.9) gets -0.2, low confidence (0.3) gets -0.05
            penalty = confidence * 0.2
            score -= penalty

        # Text length optimization (prefer medium-length segments)
        word_count = len(text.split())
        if 5 <= word_count <= 20:
            score += 0.15  # Optimal range
        elif 3 <= word_count <= 4:
            score += 0.1  # Acceptable short
        elif 21 <= word_count <= 30:
            score += 0.05  # Acceptable long
        elif word_count > 30:
            score -= 0.1  # Too long

        # Proper capitalization
        if text and text[0].isupper():
            score += 0.05

        # Proper punctuation
        if re.search(r"[.!?]$", text):
            score += 0.05

        # Timing consistency (minor factor)
        total_duration = segments[-1].end_time - segments[0].start_time
        if 1.0 <= total_duration <= 10.0:
            score += 0.05
        elif total_duration > 15.0:
            score -= 0.05  # Too long segments

        return min(1.0, max(0.0, score))
