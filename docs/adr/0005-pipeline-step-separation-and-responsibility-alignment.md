# ADR 0005: Pipeline Step Separation and Responsibility Alignment

## Status
Accepted

## Context
The current pipeline architecture conflates multiple responsibilities within the "transcription" step, creating conceptual misalignment and implementation complexity. Specifically:

**Current Pipeline**: `video → transcription → translation → tts → audio → video`

The "transcription" step (ADR 0003) currently handles:
1. Speech-to-text extraction (actual transcription)
2. Transcript text processing and enhancement
3. Speech timing alignment for translation synchronization

This creates several problems:
- **Conceptual confusion**: YouTube videos with existing transcripts still go through a "transcription" step
- **Mixed responsibilities**: The step conflates data extraction with alignment concerns
- **Inflexible abstractions**: Cannot easily handle different video sources (YouTube vs raw video)
- **Unclear boundaries**: Translation and timing alignment logic are intertwined

The core issue is that we're trying to solve two distinct problems in one step:
1. **Data extraction**: Getting transcript + timing data from various sources
2. **Alignment**: Synchronizing translated text with original speech timing

## Decision
Restructure the pipeline to separate concerns cleanly:

**New Pipeline**: `video → data_extraction → translation → alignment → tts → audio → video`

### Step Responsibilities

#### 1. Data Extraction Service (`DataExtractionService`)
**Responsibility**: Extract all available data from video sources
- YouTube: Extract transcript + timing metadata via API
- Raw video: Perform speech-to-text transcription
- Other sources: Handle format-specific extraction
- **Output**: `TimedTranscript` (text + timing metadata)

#### 2. Translation Service (`TranslationService`)
**Responsibility**: Pure text translation
- Takes clean text input
- Focuses solely on translation quality
- No timing or alignment concerns
- **Output**: `TranslatedText`

#### 3. Alignment Service (`AlignmentService`)
**Responsibility**: Synchronize translated text with original timing
- Takes `TimedTranscript` + `TranslatedText`
- Handles timing preservation challenges
- Manages segment length mismatches
- Applies speech timing optimization
- **Output**: `TimedTranslation`

### Key Interface Changes

```python
# Data Extraction
@dataclass
class TimedTranscript:
    segments: List[TranscriptSegment]
    source_type: SourceType  # YOUTUBE, TRANSCRIPTION, etc.
    timing_metadata: TimingMetadata

# Translation (simplified)
@dataclass
class TranslatedText:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str

# Alignment (specialized)
@dataclass
class TimedTranslation:
    aligned_segments: List[AlignedSegment]
    alignment_quality: AlignmentQuality
    timing_adjustments: List[TimingAdjustment]
```

### Implementation Strategy
1. **Phase 1**: Implement new `DataExtractionService` with YouTube support
2. **Phase 2**: Simplify `TranslationService` to focus on pure translation
3. **Phase 3**: Create specialized `AlignmentService` with timing logic
4. **Phase 4**: Migrate existing `HybridTranscriptProcessingService` logic to appropriate services

## Consequences

### Positive
- **Clear separation of concerns**: Each service has a single, well-defined responsibility
- **Intuitive abstractions**: YouTube videos don't need "transcription", they need "data extraction"
- **Flexible architecture**: Easy to add new video sources or translation engines
- **Maintainable code**: Timing alignment logic is isolated and specialized
- **Testable components**: Each service can be tested independently
- **Source-agnostic pipeline**: Same pipeline works for YouTube, raw video, or other sources

### Negative
- **Increased complexity**: More services to coordinate
- **Refactoring effort**: Significant changes to existing implementation
- **Interface changes**: Breaking changes to current API
- **Migration overhead**: Need to migrate existing `HybridTranscriptProcessingService`
- **Potential performance impact**: Additional service boundaries may introduce latency

### Risk Mitigation
- Implement services incrementally while maintaining backward compatibility
- Preserve existing interfaces during transition period
- Migrate alignment logic carefully to avoid timing regressions
- Add comprehensive integration tests for new pipeline flow

## Superseded Decisions
This ADR supersedes **ADR 0003: Transcript Processing Service Redesign** by fundamentally restructuring the pipeline architecture. The implemented `HybridTranscriptProcessingService` will be decomposed into the new service architecture.

## Implementation Plan
1. **Week 1**: Implement `DataExtractionService` with YouTube support
2. **Week 2**: Refactor `TranslationService` to pure translation
3. **Week 3**: Create `AlignmentService` with timing logic from existing implementation
4. **Week 4**: Integration testing and performance optimization
5. **Week 5**: Migration of existing workflows and deprecation of old interfaces

## Success Criteria
- **Conceptual clarity**: Each service has a single, obvious purpose
- **Source flexibility**: Easy to add new video sources without changing downstream services
- **Maintained quality**: Translation and timing quality equal or better than current implementation
- **Performance**: Pipeline performance within 10% of current implementation
- **Test coverage**: >90% test coverage for all new services
- **Migration success**: All existing workflows function with new architecture

## Date
2025-07-06