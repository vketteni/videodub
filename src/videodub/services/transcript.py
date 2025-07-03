"""Transcript processing service for sentence reconstruction and timing management."""

import re
import string
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import timedelta

import structlog

from ..core.models import TranscriptSegment
from ..core.exceptions import ProcessingError


logger = structlog.get_logger(__name__)


@dataclass
class SentenceSegment:
    """A complete sentence with timing information and source segments."""
    sentence_text: str
    start_time: float
    end_time: float
    source_segments: List[TranscriptSegment] = field(default_factory=list)
    confidence_score: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get the duration of this sentence."""
        return self.end_time - self.start_time
    
    def add_segment(self, segment: TranscriptSegment) -> None:
        """Add a source segment to this sentence."""
        self.source_segments.append(segment)
        # Update timing to encompass all segments
        if not self.source_segments or segment.start_time < self.start_time:
            self.start_time = segment.start_time
        if not self.source_segments or segment.end_time > self.end_time:
            self.end_time = segment.end_time


@dataclass
class TimingMap:
    """Maps original segment timing to reconstructed sentence timing."""
    original_start: float
    original_end: float
    sentence_start: float
    sentence_end: float
    sentence_index: int
    position_in_sentence: float  # 0.0 to 1.0


class TranscriptProcessingService:
    """Processes transcript segments to create natural sentence boundaries."""
    
    def __init__(self, min_sentence_length: int = 10, max_sentence_length: int = 200):
        """
        Initialize transcript processor.
        
        Args:
            min_sentence_length: Minimum characters for a complete sentence
            max_sentence_length: Maximum characters before forcing a break
        """
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        
        # Sentence ending patterns
        self.sentence_endings = re.compile(r'[.!?]+(?:\s|$)')
        self.weak_endings = re.compile(r'[,;:]+(?:\s|$)')
        
        # Common abbreviations that shouldn't end sentences
        self.abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd',
            'co', 'corp', 'gov', 'edu', 'org', 'com', 'net', 'st', 'ave', 'blvd'
        }
        
        logger.info("Transcript processor initialized", 
                   min_length=min_sentence_length, max_length=max_sentence_length)
    
    def process_transcript(self, segments: List[TranscriptSegment]) -> Tuple[List[SentenceSegment], List[TimingMap]]:
        """
        Process transcript segments into complete sentences with timing maps.
        
        Args:
            segments: List of original transcript segments
            
        Returns:
            Tuple of (sentence_segments, timing_maps)
            
        Raises:
            ProcessingError: If processing fails
        """
        if not segments:
            return [], []
        
        try:
            logger.info("Processing transcript", segment_count=len(segments))
            
            # Clean and prepare segments
            cleaned_segments = self._clean_segments(segments)
            
            # Merge segments into sentences
            sentences = self._merge_into_sentences(cleaned_segments)
            
            # Create timing maps
            timing_maps = self._create_timing_maps(segments, sentences)
            
            logger.info("Transcript processing complete", 
                       original_segments=len(segments),
                       reconstructed_sentences=len(sentences))
            
            return sentences, timing_maps
            
        except Exception as e:
            logger.error("Transcript processing failed", error=str(e))
            raise ProcessingError(f"Failed to process transcript: {str(e)}")
    
    def _clean_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Clean and normalize transcript segments."""
        cleaned = []
        
        for segment in segments:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', segment.text.strip())
            
            # Skip empty segments
            if not text:
                continue
            
            # Create cleaned segment
            cleaned_segment = TranscriptSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                text=text
            )
            cleaned.append(cleaned_segment)
        
        return cleaned
    
    def _merge_into_sentences(self, segments: List[TranscriptSegment]) -> List[SentenceSegment]:
        """Merge transcript segments into complete sentences."""
        sentences = []
        current_sentence = SentenceSegment("", 0.0, 0.0)
        
        for segment in segments:
            # Add segment text to current sentence
            if current_sentence.sentence_text:
                current_sentence.sentence_text += " " + segment.text
            else:
                current_sentence.sentence_text = segment.text
                current_sentence.start_time = segment.start_time
            
            current_sentence.end_time = segment.end_time
            current_sentence.add_segment(segment)
            
            # Check if we should end the current sentence
            should_end = self._should_end_sentence(current_sentence.sentence_text)
            
            if should_end:
                # Finalize current sentence
                current_sentence.sentence_text = current_sentence.sentence_text.strip()
                current_sentence.confidence_score = self._calculate_confidence(current_sentence.sentence_text)
                sentences.append(current_sentence)
                
                # Start new sentence
                current_sentence = SentenceSegment("", 0.0, 0.0)
        
        # Handle remaining text
        if current_sentence.sentence_text.strip():
            current_sentence.sentence_text = current_sentence.sentence_text.strip()
            current_sentence.confidence_score = self._calculate_confidence(current_sentence.sentence_text)
            sentences.append(current_sentence)
        
        return sentences
    
    def _should_end_sentence(self, text: str) -> bool:
        """Determine if the current text should end a sentence."""
        text = text.strip()
        
        # Force break if too long
        if len(text) > self.max_sentence_length:
            return True
        
        # Don't break if too short
        if len(text) < self.min_sentence_length:
            return False
        
        # Check for strong sentence endings
        if self.sentence_endings.search(text):
            # Check if it's a false positive (abbreviation)
            words = text.lower().split()
            if words:
                last_word = words[-1].rstrip(string.punctuation)
                if last_word in self.abbreviations:
                    return False
            return True
        
        return False
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for sentence completeness."""
        # Base confidence
        confidence = 0.5
        
        # Bonus for proper sentence endings
        if self.sentence_endings.search(text):
            confidence += 0.3
        
        # Bonus for reasonable length
        if self.min_sentence_length <= len(text) <= self.max_sentence_length:
            confidence += 0.2
        
        # Bonus for starting with capital letter
        if text and text[0].isupper():
            confidence += 0.1
        
        # Penalty for very short or very long
        if len(text) < self.min_sentence_length:
            confidence -= 0.2
        elif len(text) > self.max_sentence_length:
            confidence -= 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _create_timing_maps(self, original_segments: List[TranscriptSegment], 
                           sentences: List[SentenceSegment]) -> List[TimingMap]:
        """Create timing maps between original segments and reconstructed sentences."""
        timing_maps = []
        
        for sentence_idx, sentence in enumerate(sentences):
            for source_segment in sentence.source_segments:
                # Calculate position within sentence
                sentence_duration = sentence.duration
                segment_start_offset = source_segment.start_time - sentence.start_time
                position_in_sentence = segment_start_offset / sentence_duration if sentence_duration > 0 else 0.0
                
                timing_map = TimingMap(
                    original_start=source_segment.start_time,
                    original_end=source_segment.end_time,
                    sentence_start=sentence.start_time,
                    sentence_end=sentence.end_time,
                    sentence_index=sentence_idx,
                    position_in_sentence=position_in_sentence
                )
                timing_maps.append(timing_map)
        
        return timing_maps
    
    def reconstruct_timing(self, sentence_audio_duration: float, 
                          timing_maps: List[TimingMap], sentence_index: int) -> Dict[str, float]:
        """
        Reconstruct timing for a translated sentence audio.
        
        Args:
            sentence_audio_duration: Duration of the generated sentence audio
            timing_maps: List of timing maps for reconstruction
            sentence_index: Index of the sentence being processed
            
        Returns:
            Dictionary with timing information for synchronization
        """
        # Get timing maps for this sentence
        sentence_maps = [tm for tm in timing_maps if tm.sentence_index == sentence_index]
        
        if not sentence_maps:
            return {"start": 0.0, "end": sentence_audio_duration, "segments": []}
        
        # Calculate original sentence timing
        original_start = min(tm.original_start for tm in sentence_maps)
        original_end = max(tm.original_end for tm in sentence_maps)
        original_duration = original_end - original_start
        
        # Calculate scaling factor
        scale_factor = sentence_audio_duration / original_duration if original_duration > 0 else 1.0
        
        # Create segment timing information
        segment_timings = []
        for tm in sentence_maps:
            # Scale the timing proportionally
            new_start = (tm.original_start - original_start) * scale_factor
            new_end = (tm.original_end - original_start) * scale_factor
            
            segment_timings.append({
                "original_start": tm.original_start,
                "original_end": tm.original_end,
                "new_start": new_start,
                "new_end": new_end,
                "position": tm.position_in_sentence
            })
        
        return {
            "sentence_start": original_start,
            "sentence_end": original_end,
            "audio_duration": sentence_audio_duration,
            "scale_factor": scale_factor,
            "segments": segment_timings
        }


def create_transcript_processing_service(**kwargs) -> TranscriptProcessingService:
    """Factory function to create a transcript processing service."""
    return TranscriptProcessingService(**kwargs)