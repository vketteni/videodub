# ADR 0002: Real-Time Cost Tracking Integration

## Status
Accepted

## Context
API costs for translation and TTS services can accumulate quickly, especially with:
- Multiple OpenAI GPT model options with different pricing
- Variable text lengths and character counts for TTS
- Batch processing of multiple videos
- Need for cost optimization and budgeting

Users need:
- Real-time cost visibility during processing
- Accurate cost breakdowns by service and model
- Economic model comparison capabilities
- Session-based cost summaries

## Decision
Implement a **Global Cost Tracker** with the following architecture:

### Integration Points
1. **Translation Service**: Track token usage for each GPT model call
2. **TTS Service**: Track character usage for audio generation
3. **Pipeline Orchestrator**: Capture session summaries and reset tracking
4. **Cost Calculation**: Real-time pricing calculations using current API rates

### Implementation Strategy
```python
# Global cost tracker instance
cost_tracker = CostTracker()

# Service integration
translation_service.track_usage(model, tokens)
tts_service.track_usage(characters)

# Pipeline integration
result.cost_summary = get_session_cost_summary()
reset_global_cost_tracker()  # For new sessions
```

### Data Structure
```python
@dataclass
class CostBreakdown:
    model: str
    input_tokens: int
    output_tokens: int
    characters: int
    input_cost: float
    output_cost: float
    audio_cost: float
    total_cost: float
    requests: int
```

## Consequences

### Positive
- **Transparency**: Users see exact costs in real-time
- **Optimization**: Easy comparison between models and engines
- **Budgeting**: Predictable cost estimation for batch processing
- **Debugging**: Cost breakdowns help identify expensive operations
- **Economic Testing**: `quick_test.py` can compare model economics

### Negative
- **Coupling**: Services need to be aware of cost tracking
- **Maintenance**: Pricing models need updates when APIs change
- **Storage**: Additional data structures and memory usage
- **Complexity**: More code paths and potential failure points

### Mitigations
- Keep cost tracking optional and non-blocking
- Provide fallback behavior if cost tracking fails
- Update pricing models regularly
- Make cost tracking easily disableable for testing

## Implementation Notes
- Cost tracking implemented in `utils/cost_tracking.py`
- Global tracker pattern to avoid passing tracker through all method calls
- Pricing models stored as constants, easily updatable
- Session-based tracking allows per-video cost analysis

## Alternatives Considered
1. **No Cost Tracking**: Simplest but provides no visibility
2. **External Cost Tracking**: Separate service, but adds complexity
3. **Database Tracking**: Persistent storage, but overkill for current needs

## Date
2025-07-03