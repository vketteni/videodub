# ADR-0006: Speech Timing Model Redesign

## Status
Proposed

## Context

The current pipeline transforms YouTube's `start + duration` timing data to `start + end` timing by calculating `end_time = start_time + duration`. This approach has fundamental issues:

### Current Problems
1. **Overlapping Segments**: YouTube durations represent subtitle display time, not speech time, causing massive overlaps (4+ segments playing simultaneously)
2. **Unnatural Timing**: "[Music]" segments overlap with speech, single speakers appear to say multiple phrases simultaneously  
3. **Alignment Difficulties**: Translation alignment struggles with overlapping source timing
4. **Audio Generation Issues**: TTS systems can't handle overlapping speech segments

### YouTube's Timing Model
Analysis of raw data reveals YouTube provides:
- `start`: Actual speech start time (reliable)
- `duration`: Subtitle display duration (optimized for readability, not speech timing)

Example problematic timing:
```
Segment 1: start=0.12, duration=2.44 → 0.12→2.56 "[Music]"
Segment 2: start=0.32, duration=4.4  → 0.32→4.72 "As the festivities wound down..."
Segment 3: start=2.56, duration=4.08 → 2.56→6.64 "moment to speak to Daniela..."
```

## Decision

We will implement **Model A: Speech Estimation + Natural Pause Spacing**:

1. **Estimate actual speech duration** using sophisticated heuristics based on text content
2. **Preserve YouTube's start times** as they reflect natural conversation rhythm
3. **Let pauses emerge naturally** from the gap between estimated speech end and next segment start
4. **Separate speech and pause handling** in the audio pipeline

### Timing Model
```
Segment: [speech_start → speech_end] + [pause_until_next_start]
Where: speech_end = speech_start + estimated_speech_duration(text)
       pause_duration = next_segment.start - current_segment.speech_end
```

### Speech Duration Estimation Strategy
- Base calculation on syllable count (~4.5 syllables/second)
- Adjust for text complexity, numbers, emotional content
- Include internal punctuation micro-pauses
- Apply reasonable bounds (0.3s - 10.0s)

## Consequences

### Positive
- **Natural Speech Timing**: Eliminates impossible overlapping speech
- **Improved Translation Quality**: Alignment works with realistic timing
- **Better Audio Generation**: TTS generates natural-length audio
- **Preserves Conversation Flow**: Natural pauses and rhythm maintained
- **Flexible Audio Pipeline**: Can handle gaps with silence, ambient sound, or cross-fades

### Negative
- **Implementation Complexity**: Requires speech duration estimation algorithms
- **Potential Timing Inaccuracy**: Heuristics may not perfectly match actual speech
- **Pipeline Changes**: Audio assembly must handle speech/pause separation

### Risks
- **Estimation Errors**: Poor heuristics could create unnatural timing
- **Edge Cases**: Special content (music, sound effects) may need custom handling

## Alternatives Considered

### Alternative 1: Gap-Free Sequential
- `end_time = next_segment.start_time`
- Simpler but loses natural pause information

### Alternative 2: Keep YouTube Duration
- Use YouTube's timing as-is
- Doesn't solve fundamental overlap problems

### Alternative 3: Speech + Embedded Pauses  
- Include pause time within each segment
- Less flexible for audio processing

## Implementation

This ADR will be implemented through:
1. Enhanced speech duration estimation in data extraction
2. Modified timing models in core pipeline
3. Updated audio assembly to handle speech/pause separation
4. Alignment service updates for new timing model

## Date
2025-07-07

## References
- Pipeline Architecture: `docs/architecture/current_architecture.md`
- Related: ADR-0001 (Pipeline Architecture), ADR-0005 (Service Separation)