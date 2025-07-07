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

### YouTube's Timing Model - Critical Distinction
Analysis of raw data reveals YouTube provides:
- `start`: **Actual speech start time** (reliable - represents when words are spoken)
- `duration`: **Subtitle display duration** (optimized for readability, NOT speech timing)

**The Core Misunderstanding**: Our pipeline incorrectly treats `duration` as speech duration, when it actually represents how long the subtitle should remain visible on screen for optimal reading experience.

Example problematic timing:
```
YouTube Data:
Segment 1: start=0.12, duration=2.44, text="[Music]"
Segment 2: start=0.32, duration=4.4,  text="As the festivities wound down..."
Segment 3: start=2.56, duration=4.08, text="moment to speak to Daniela..."

Current Pipeline (WRONG):
Segment 1: speech_time=0.12→2.56 "[Music]"           (2.44s speech)
Segment 2: speech_time=0.32→4.72 "As the festivities..." (4.4s speech)  
Segment 3: speech_time=2.56→6.64 "moment to speak..."   (4.08s speech)
→ Result: Massive overlaps, 4+ segments "speaking" simultaneously

Correct Model (ADR-0006):
Segment 1: speech_time=0.12→0.32 "[Music]"           (0.2s speech, 0s pause)
Segment 2: speech_time=0.32→2.56 "As the festivities..." (2.24s speech, 0s pause)
Segment 3: speech_time=2.56→4.80 "moment to speak..."   (2.24s speech, pause until next)
→ Result: No overlaps, natural speech timing
```

**Key Insight**: YouTube's `start` times represent natural conversation rhythm and speech timing. The `duration` field is purely for subtitle display optimization and should be **completely ignored** for speech timing purposes.

## Decision

We will implement **Model A: Speech Estimation + Natural Pause Spacing**:

1. **Estimate actual speech duration** using sophisticated heuristics based on text content
2. **Preserve YouTube's start times** as they reflect natural conversation rhythm
3. **Let pauses emerge naturally** from the gap between estimated speech end and next segment start
4. **Separate speech and pause handling** in the audio pipeline

### New Timing Model
```
Input (YouTube):  {start, duration, text}
Ignore:          duration (subtitle display time)
Use:             start (actual speech start time)
Estimate:        speech_duration = estimate_from_text(text)

New Timing:
speech_start = youtube.start                    (trusted)
speech_end = speech_start + estimated_speech_duration(text)  (calculated)
pause_duration = next_segment.start - current_segment.speech_end  (natural)

Segment: [speech_start → speech_end] + [pause_until_next_start]
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

This ADR maintains **separation of concerns** while correcting the YouTube data misinterpretation:

### Pipeline Design
```
DataExtraction: YouTube {start, duration, text}
                ↓ (PRESERVE all data, label correctly)
                TranscriptSegment(
                  start_time=start,
                  subtitle_display_duration=duration,  # preserved but labeled correctly
                  text=text,
                  end_time=None  # incomplete timing
                )
                ↓
Translation:    List[TranscriptSegment] → List[str] (translated_texts)
                ↓
Alignment:      TimedTranscript + List[str] 
                ↓ (APPLY speech estimation here - proper service responsibility)
                Complete timing: end_time = start_time + estimated_speech_duration(text)
                ↓
                TimedTranslation (with proper speech timing)
                ↓
TTS/Audio/Video: (works with corrected timing)
```

### Implementation Components
1. **Enhanced TranscriptSegment Model**: Preserve all YouTube data with correct labels
2. **Speech Duration Estimation in AlignmentService**: Timing logic stays in alignment service
3. **Modified data flow**: Data extraction provides incomplete timing, Alignment completes it
4. **Updated audio assembly**: Handle speech/pause separation

### Design Principles
- **Preserve all data**: Keep YouTube's duration as `subtitle_display_duration` for future use
- **Clear labeling**: Never ambiguous about what duration represents  
- **Separation of concerns**: Speech estimation stays in Alignment service
- **Incomplete → Complete**: Data extraction provides incomplete timing, Alignment completes it

## Future Work
This timing model change reveals naming inconsistencies in our models. A follow-up refactoring should address misleading names like `TimedTranscript` and `TimedTranslation` to better reflect the actual data flow where timing is incomplete from extraction and only properly established during alignment.

## Date
2025-07-07

## References
- Pipeline Architecture: `docs/architecture/current_architecture.md`
- Related: ADR-0001 (Pipeline Architecture), ADR-0005 (Service Separation)