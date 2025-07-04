"""Transcript processing step evaluator."""

import logging
import re
from typing import Any, Dict, List

from ..evaluators.base import (
    EvaluationConfig,
    EvaluationDataset,
    EvaluationResult,
    PipelineStepEvaluator,
    run_evaluation_sample,
)

# Set up logger first
logger = logging.getLogger(__name__)

# Import the actual transcript processing service
try:
    # Try simple import first
    import sys
    from pathlib import Path
    
    # Add src to path temporarily for import
    src_path = str(Path(__file__).parent.parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        
    # Try importing models and service directly
    from videodub.core.models import (
        ProcessedSegment,
        ProcessingMode, 
        TranscriptSegment,
    )
    
    # Try to import the real service
    try:
        from videodub.services.transcript import HybridTranscriptProcessingService
        logger.info("✅ Successfully imported REAL HybridTranscriptProcessingService")
        USING_REAL_SERVICE = True
    except ImportError as service_import_error:
        logger.warning(f"Could not import real service: {service_import_error}")
        
        # Fallback to improved mock with sentence processing
        class ImprovedMockTranscriptProcessingService:
            """Improved mock with sentence-aware processing."""
            
            async def process_transcript(self, segments, processing_mode):
                """Mock processing with sentence boundary detection."""
                if not segments:
                    return []
                
                processed_segments = []
                current_group = []
                sequence_num = 0
                
                for i, segment in enumerate(segments):
                    current_group.append(segment)
                    
                    # Check if this segment ends a sentence or we should break
                    should_break = (
                        self._ends_sentence(segment.text) or
                        len(current_group) >= 3 or  # Max 3 segments per group
                        i == len(segments) - 1  # Last segment
                    )
                    
                    if should_break:
                        # Create processed segment from current group
                        merged_text = " ".join(seg.text for seg in current_group)
                        merged_text = self._fix_text(merged_text)
                        
                        processed_segment = ProcessedSegment(
                            merged_segments=current_group.copy(),
                            processed_text=merged_text,
                            processing_mode=processing_mode,
                            sequence_number=sequence_num,
                            original_indices=list(range(i - len(current_group) + 1, i + 1)),
                            is_sentence_complete=self._ends_sentence(merged_text),
                            context_quality_score=0.8,
                            ready_for_translation=True
                        )
                        processed_segments.append(processed_segment)
                        
                        # Reset for next group
                        current_group = []
                        sequence_num += 1
                
                return processed_segments
            
            def _ends_sentence(self, text: str) -> bool:
                """Check if text ends a sentence."""
                text = text.strip()
                return bool(text and (
                    text.endswith('.') or 
                    text.endswith('!') or 
                    text.endswith('?') or
                    text.endswith(':')
                ))
            
            def _fix_text(self, text: str) -> str:
                """Basic text cleanup."""
                text = text.strip()
                if text:
                    text = text[0].upper() + text[1:]  # Capitalize first letter
                    if not self._ends_sentence(text):
                        text += '.'  # Add period if missing
                return text
        
        HybridTranscriptProcessingService = ImprovedMockTranscriptProcessingService
        logger.info("Using improved mock transcript processing service with sentence awareness")
        USING_REAL_SERVICE = False
    
except Exception as e:
    logger.warning(f"Could not import transcript processing models: {e}")
    logger.warning("Using complete mock objects for testing")
    # Fallback for testing
    from unittest.mock import Mock
    TranscriptSegment = Mock
    ProcessedSegment = Mock
    ProcessingMode = Mock
    HybridTranscriptProcessingService = Mock


class TranscriptProcessingEvaluator(PipelineStepEvaluator):
    """Evaluator for transcript processing pipeline step."""
    
    def __init__(self):
        """Initialize transcript processing evaluator."""
        super().__init__("transcript_processing")
        self.service = HybridTranscriptProcessingService()
        
    async def evaluate_step(
        self, 
        dataset: EvaluationDataset, 
        config: EvaluationConfig
    ) -> List[EvaluationResult]:
        """Run evaluation on transcript processing step with given dataset."""
        results = []
        
        # Limit samples if specified
        samples = dataset.input_samples
        if config.max_samples:
            samples = samples[:config.max_samples]
            
        for i, sample in enumerate(samples):
            self.logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.get('description', 'Unknown')}")
            
            # Run evaluation on single sample
            result = await run_evaluation_sample(sample, self, config)
            result.dataset_name = dataset.name
            
            results.append(result)
            
        return results
    
    async def _process_sample(self, input_data: Any, config: EvaluationConfig) -> Any:
        """Process a single sample through the transcript processing service."""
        # Convert input data to TranscriptSegment objects
        segments = []
        for segment_data in input_data.get("segments", []):
            segment = TranscriptSegment(
                start_time=segment_data["start"],
                end_time=segment_data["end"],
                text=segment_data["text"]
            )
            segments.append(segment)
        
        # Determine processing mode
        processing_mode = ProcessingMode.HYBRID
        if config.processing_mode:
            if config.processing_mode.upper() == "RULE_BASED":
                processing_mode = ProcessingMode.RULE_BASED
            elif config.processing_mode.upper() == "AI_ENHANCED":
                processing_mode = ProcessingMode.AI_ENHANCED
            elif config.processing_mode.upper() == "HYBRID":
                processing_mode = ProcessingMode.HYBRID
        
        # Process segments
        processed_segments = await self.service.process_transcript(segments, processing_mode)
        
        # Convert back to serializable format
        output_data = {
            "processed_segments": [
                {
                    "merged_segments": [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "text": seg.text
                        } for seg in ps.merged_segments
                    ],
                    "processed_text": ps.processed_text,
                    "processing_mode": ps.processing_mode.value if hasattr(ps.processing_mode, 'value') else str(ps.processing_mode),
                    "sequence_number": ps.sequence_number,
                    "original_indices": ps.original_indices,
                    "quality_score": getattr(ps, 'context_quality_score', 0.0),
                    "processing_metadata": getattr(ps, 'enhancement_metadata', {})
                } for ps in processed_segments
            ]
        }
        
        return output_data
    
    def calculate_quality_score(self, input_data: Any, output_data: Any) -> float:
        """Calculate quality score for transcript processing evaluation."""
        if not output_data or "processed_segments" not in output_data:
            return 0.0
        
        processed_segments = output_data["processed_segments"]
        if not processed_segments:
            return 0.0
            
        # Calculate composite quality score
        scores = []
        
        for segment in processed_segments:
            segment_score = 0.0
            
            # Check if segment has quality_score from the service
            if "quality_score" in segment and segment["quality_score"] is not None:
                segment_score = segment["quality_score"]
            else:
                # Fallback quality assessment
                segment_score = self._assess_segment_quality(segment)
            
            scores.append(segment_score)
        
        # Return average quality score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_segment_quality(self, segment: Dict[str, Any]) -> float:
        """Assess quality of a single processed segment."""
        text = segment.get("processed_text", "")
        if not text:
            return 0.0
            
        score = 0.0
        
        # Check for complete sentences (ends with proper punctuation)
        if re.search(r'[.!?]$', text.strip()):
            score += 0.3
        
        # Check for proper capitalization
        if text and text[0].isupper():
            score += 0.2
            
        # Check for reasonable length (not too short, not too long)
        word_count = len(text.split())
        if 5 <= word_count <= 50:
            score += 0.2
        elif word_count > 1:
            score += 0.1
            
        # Check for coherence (no obvious fragments)
        if not re.search(r'\b(um|uh|er|ah)\b', text.lower()):
            score += 0.1
            
        # Check for proper spacing and no double spaces
        if not re.search(r'  +', text):
            score += 0.1
            
        # Check for technical term capitalization
        technical_terms = ['API', 'REST', 'JSON', 'SQL', 'GPU', 'CPU', 'AI', 'ML']
        for term in technical_terms:
            if term.lower() in text.lower() and term in text:
                score += 0.05
                break
                
        # Ensure score is between 0 and 1
        return min(score, 1.0)
    
    def extract_metrics(self, input_data: Any, output_data: Any) -> Dict[str, float]:
        """Extract transcript processing specific metrics."""
        metrics = {}
        
        if not output_data or "processed_segments" not in output_data:
            return metrics
            
        input_segments = input_data.get("segments", [])
        processed_segments = output_data["processed_segments"]
        
        # Basic metrics
        metrics["input_segment_count"] = len(input_segments)
        metrics["output_segment_count"] = len(processed_segments)
        metrics["segment_reduction_ratio"] = (
            len(input_segments) - len(processed_segments)
        ) / len(input_segments) if input_segments else 0.0
        
        # Text metrics
        input_text = " ".join(seg.get("text", "") for seg in input_segments)
        output_text = " ".join(seg.get("processed_text", "") for seg in processed_segments)
        
        metrics["input_word_count"] = len(input_text.split())
        metrics["output_word_count"] = len(output_text.split())
        metrics["word_preservation_ratio"] = (
            metrics["output_word_count"] / metrics["input_word_count"]
        ) if metrics["input_word_count"] > 0 else 0.0
        
        # Quality metrics
        complete_sentences = sum(
            1 for seg in processed_segments 
            if re.search(r'[.!?]$', seg.get("processed_text", "").strip())
        )
        metrics["complete_sentence_ratio"] = (
            complete_sentences / len(processed_segments)
        ) if processed_segments else 0.0
        
        # Capitalization metrics
        properly_capitalized = sum(
            1 for seg in processed_segments 
            if seg.get("processed_text", "") and seg["processed_text"][0].isupper()
        )
        metrics["proper_capitalization_ratio"] = (
            properly_capitalized / len(processed_segments)
        ) if processed_segments else 0.0
        
        # Timing preservation
        input_duration = sum(
            seg.get("end", 0) - seg.get("start", 0) for seg in input_segments
        )
        output_duration = sum(
            seg_data.get("end", 0) - seg_data.get("start", 0)
            for seg in processed_segments
            for seg_data in seg.get("merged_segments", [])
        )
        metrics["timing_preservation_ratio"] = (
            output_duration / input_duration
        ) if input_duration > 0 else 0.0
        
        # Speaker preservation
        input_speakers = set(seg.get("speaker", "") for seg in input_segments)
        output_speakers = set(
            seg_data.get("speaker", "")
            for seg in processed_segments
            for seg_data in seg.get("merged_segments", [])
        )
        metrics["speaker_preservation_ratio"] = (
            len(output_speakers & input_speakers) / len(input_speakers)
        ) if input_speakers else 0.0
        
        return metrics
    
    def _generate_recommendations(
        self, 
        configurations: List[str], 
        quality_comparison: Dict[str, float],
        performance_comparison: Dict[str, float],
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate transcript processing specific recommendations."""
        recommendations = super()._generate_recommendations(
            configurations, quality_comparison, performance_comparison, metrics_comparison
        )
        
        # Add transcript-specific recommendations
        for config_name, metrics in metrics_comparison.items():
            # Check segment reduction
            if metrics.get("segment_reduction_ratio", 0) > 0.7:
                recommendations.append(
                    f"**{config_name}**: High segment reduction ({metrics['segment_reduction_ratio']:.1%}) - "
                    "good for coherence but verify important content isn't lost"
                )
            
            # Check sentence completion
            if metrics.get("complete_sentence_ratio", 0) < 0.8:
                recommendations.append(
                    f"**{config_name}**: Low sentence completion rate ({metrics['complete_sentence_ratio']:.1%}) - "
                    "may need improved sentence boundary detection"
                )
            
            # Check capitalization
            if metrics.get("proper_capitalization_ratio", 0) < 0.9:
                recommendations.append(
                    f"**{config_name}**: Inconsistent capitalization ({metrics['proper_capitalization_ratio']:.1%}) - "
                    "consider improving text normalization"
                )
        
        return recommendations
    
    def _generate_step_specific_comparison(self, input_data: Any, output_data: Any) -> str:
        """Generate transcript-specific visual comparison."""
        if not isinstance(input_data, dict) or not isinstance(output_data, dict):
            return super()._generate_step_specific_comparison(input_data, output_data)
        
        comparison = ""
        
        # Extract input segments
        input_segments = input_data.get("segments", [])
        processed_segments = output_data.get("processed_segments", [])
        
        if input_segments:
            comparison += f"**Before ({len(input_segments)} segments):**\n"
            for i, seg in enumerate(input_segments, 1):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "")
                speaker = seg.get("speaker", "")
                speaker_info = f" [{speaker}]" if speaker else ""
                comparison += f"{i}. [{start:.1f}-{end:.1f}s]{speaker_info} \"{text}\"\n"
            
            comparison += "\n"
        
        if processed_segments:
            comparison += f"**After ({len(processed_segments)} segments):**\n"
            for i, ps in enumerate(processed_segments, 1):
                processed_text = ps.get("processed_text", "")
                merged_segs = ps.get("merged_segments", [])
                processing_mode = ps.get("processing_mode", "unknown")
                sequence_number = ps.get("sequence_number", "")
                quality_score = ps.get("quality_score", "")
                
                if merged_segs:
                    start = min(seg.get("start_time", 0) for seg in merged_segs)
                    end = max(seg.get("end_time", 0) for seg in merged_segs)
                    timing_info = f"[{start:.1f}-{end:.1f}s]"
                    
                    # Show which original segments were merged
                    original_count = len(merged_segs)
                    if original_count > 1:
                        merged_info = f" (merged {original_count} segments)"
                    else:
                        merged_info = ""
                else:
                    timing_info = "[timing unknown]"
                    merged_info = ""
                
                # Add quality info if available
                quality_info = ""
                if quality_score:
                    quality_info = f" [Q: {quality_score:.2f}]"
                
                comparison += f"{i}. {timing_info}{merged_info}{quality_info} \"{processed_text}\"\n"
                
                # Show the original segments that were merged (if more than 1)
                if merged_segs and len(merged_segs) > 1:
                    comparison += f"   └─ Merged from segments: "
                    for j, seg in enumerate(merged_segs):
                        if j > 0:
                            comparison += " + "
                        seg_start = seg.get("start_time", 0)
                        seg_end = seg.get("end_time", 0) 
                        seg_text = seg.get("text", "")[:30]
                        if len(seg.get("text", "")) > 30:
                            seg_text += "..."
                        comparison += f"[{seg_start:.1f}-{seg_end:.1f}s: \"{seg_text}\"]"
                    comparison += "\n"
            
            comparison += "\n"
        
        # Add change analysis
        if input_segments and processed_segments:
            changes = []
            
            # Check for merging
            if len(processed_segments) < len(input_segments):
                reduction = len(input_segments) - len(processed_segments)
                changes.append(f"✅ Merged {reduction} fragments into coherent segments")
            
            # Check for capitalization
            input_texts = [seg.get("text", "") for seg in input_segments]
            output_texts = [ps.get("processed_text", "") for ps in processed_segments]
            
            input_capitalized = sum(1 for text in input_texts if text and text[0].isupper())
            output_capitalized = sum(1 for text in output_texts if text and text[0].isupper())
            
            if output_capitalized > input_capitalized:
                changes.append("✅ Improved capitalization")
            
            # Check for sentence completion
            input_complete = sum(1 for text in input_texts if text.strip().endswith(('.', '!', '?')))
            output_complete = sum(1 for text in output_texts if text.strip().endswith(('.', '!', '?')))
            
            if output_complete > input_complete:
                changes.append("✅ Added proper sentence endings")
            
            # Word preservation
            input_words = sum(len(text.split()) for text in input_texts)
            output_words = sum(len(text.split()) for text in output_texts)
            preservation_ratio = output_words / input_words if input_words > 0 else 0
            
            if preservation_ratio >= 0.95:
                changes.append("✅ Preserved all content")
            elif preservation_ratio >= 0.85:
                changes.append("⚠️ Minor content changes")
            else:
                changes.append("❌ Significant content loss")
            
            if changes:
                comparison += "**Changes Made:**\n"
                for change in changes:
                    comparison += f"- {change}\n"
                comparison += "\n"
        
        return comparison