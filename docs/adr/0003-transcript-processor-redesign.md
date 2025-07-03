# ADR 0003: Transcript Processing Service Redesign

## Status
Proposed

## Context
The current `TranscriptProcessingService` implementation needs reimplementation due to:
- Suboptimal sentence reconstruction logic
- Poor handling of transcript timing and segmentation
- Lack of context awareness for translation quality
- Missing error handling for malformed transcripts
- No support for different transcript formats

Current pipeline shows step 2 as "NEEDS_REIMPLEMENTATION" in workflow documentation.

## Problem Statement
Video transcripts often contain:
- Incomplete sentences split across segments
- Poor punctuation and capitalization
- Timing misalignments
- Multiple speakers without clear boundaries
- Various formats from different video sources

This leads to:
- Poor translation quality due to fragmented context
- Unnatural TTS output from incomplete sentences
- Timing synchronization issues in final video

## Decision Options

### Option 1: Rule-Based Sentence Reconstruction
**Approach**: Use linguistic rules and heuristics
- Sentence boundary detection
- Capitalization and punctuation correction
- Timing-aware segmentation

**Pros**: Predictable, fast, language-agnostic
**Cons**: Limited accuracy, hard to maintain rules

### Option 2: AI-Powered Context Reconstruction
**Approach**: Use OpenAI GPT for intelligent sentence reconstruction
- Context-aware sentence completion
- Speaker identification
- Natural language flow optimization

**Pros**: High quality, context-aware, adaptive
**Cons**: Additional API costs, latency, complexity

### Option 3: Hybrid Approach (Recommended)
**Approach**: Combine rule-based preprocessing with AI enhancement
- Fast rule-based initial processing
- AI enhancement for complex cases
- Configurable quality vs. cost trade-offs

## Recommended Implementation

```python
class TranscriptProcessingService:
    """
    Hybrid transcript processing service with rule-based and AI-powered modes.
    """
    
    async def process_transcript(
        self, 
        segments: List[TranscriptSegment],
        mode: ProcessingMode = ProcessingMode.HYBRID
    ) -> List[ProcessedSegment]:
        # Phase 1: Rule-based preprocessing
        preprocessed = self._rule_based_preprocessing(segments)
        
        # Phase 2: AI enhancement (if enabled)
        if mode in [ProcessingMode.AI_ENHANCED, ProcessingMode.HYBRID]:
            enhanced = await self._ai_enhance_segments(preprocessed)
            return enhanced
            
        return preprocessed
```

### Processing Phases
1. **Rule-Based Preprocessing**:
   - Merge short segments
   - Fix basic punctuation
   - Detect sentence boundaries
   - Preserve timing information

2. **AI Enhancement** (Optional):
   - Complete fragmented sentences
   - Improve context flow
   - Optimize for translation quality

3. **Quality Validation**:
   - Verify segment integrity
   - Check timing consistency
   - Ensure translation readiness

## Consequences

### Positive
- **Flexibility**: Multiple processing modes for different use cases
- **Quality**: Better translation input leads to better final output
- **Performance**: Fast rule-based option for cost-sensitive scenarios
- **Maintainability**: Clear separation between rule-based and AI logic

### Negative
- **Complexity**: More sophisticated implementation required
- **Cost**: AI enhancement adds to API usage
- **Development Time**: Significant effort to implement properly

### Risk Mitigation
- Implement rule-based mode first for immediate functionality
- Add AI enhancement as optional feature
- Provide clear cost/quality trade-off documentation
- Include comprehensive testing with various transcript formats

## Implementation Plan
1. **Phase 1**: Implement rule-based processor
2. **Phase 2**: Add AI enhancement capabilities  
3. **Phase 3**: Optimize hybrid mode selection logic
4. **Phase 4**: Add comprehensive testing and validation

## Success Criteria
- Improved translation quality scores
- Reduced fragmented sentences in output
- Maintained or improved processing speed
- Clear cost vs. quality trade-offs

## Date
2025-07-03