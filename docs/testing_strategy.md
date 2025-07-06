# Testing Strategy

This document outlines the comprehensive testing strategy for the videodub project, covering all test categories, execution environments, and best practices.

## Overview

The videodub project employs a multi-layered testing approach with environment-aware configuration, ensuring both rapid development feedback and thorough quality validation.

### Test Architecture Philosophy

- **Environment-Adaptive**: Tests automatically adjust based on CI vs local development environments
- **Realistic Mocking**: Mock services use realistic delays and behavior patterns
- **Performance-Aware**: Tests validate both functional correctness and performance characteristics
- **Scalable Design**: Test parameters scale with environment capabilities

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation with minimal dependencies.

**Characteristics**:
- Fast execution (< 1 second per test)
- No external dependencies
- Mock all external services
- Focus on business logic and edge cases

**Current Coverage**:
- **Models** (`test_models.py`): Domain model validation, dataclass behavior, enum handling
- **Data Extraction** (`test_data_extraction.py`): YouTube API integration, transcript parsing, metadata extraction
- **Translation Service** (`test_translation_service.py`): OpenAI GPT integration, batch processing, error handling
- **Alignment Service** (`test_alignment_service.py`): Timing synchronization strategies, A/B testing, quality metrics
- **Pipeline Orchestration** (`test_new_pipeline.py`): Service coordination, error propagation, configuration handling

**Execution**:
```bash
# Run all unit tests
make test-fast
poetry run pytest tests/unit/ -v

# Run specific test file
poetry run pytest tests/unit/test_models.py -v
```

**Key Patterns**:
- Use `@pytest.fixture` for test data setup
- Mock external services with `AsyncMock` for async operations
- Test both success and failure scenarios
- Validate domain model constraints and business rules

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and end-to-end workflows.

**Characteristics**:
- Moderate execution time (1-10 minutes)
- Test service integration points
- Use realistic mock data and timing
- Validate data flow between components

**Current Coverage**:

#### Pipeline Integration (`test_pipeline_integration.py`)
- End-to-end video processing workflows
- Service coordination and dependency injection
- Error handling and recovery
- Configuration validation

#### Service Integration (`test_service_integration.py`)
- Inter-service communication
- Data transformation between services
- Interface compliance validation
- Concurrent operation testing

#### Performance Integration (`test_performance_integration.py`)
**See detailed section below**

#### Performance Optimizations (`test_performance_optimizations.py`)
- Optimized service implementations
- Batch processing efficiency
- Concurrent processing validation
- Memory usage optimization

**Execution**:
```bash
# Run all integration tests
make test-integration
poetry run pytest tests/integration/ -v

# Run specific integration test
poetry run pytest tests/integration/test_pipeline_integration.py -v
```

### 3. Performance Tests

**Purpose**: Validate system performance characteristics and identify bottlenecks.

#### Environment-Based Configuration

Performance tests automatically adapt based on environment:

```python
TEST_CONFIG = {
    "CI": {
        "segment_count": 20,
        "data_extraction_delay": 0.01,
        "translation_delay_per_text": 0.005,
        "alignment_delay_per_segment": 0.002,
        "tts_delay_per_audio": 0.01,
        "scalability_segment_count": 100,
        "resource_cleanup_iterations": 2,
        "concurrent_test_count": 2,
    },
    "LOCAL": {
        "segment_count": 100,
        "data_extraction_delay": 0.1,
        "translation_delay_per_text": 0.05,
        "alignment_delay_per_segment": 0.02,
        "tts_delay_per_audio": 0.1,
        "scalability_segment_count": 1000,
        "resource_cleanup_iterations": 3,
        "concurrent_test_count": 3,
    },
}
```

Environment detection:
```python
TEST_ENV = "CI" if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS") else "LOCAL"
```

#### Test Categories

**Fast Performance Tests** (`@pytest.mark.fast`):
- **Pipeline Processing Time**: Basic end-to-end timing validation
- **Concurrent Processing**: Multi-video parallel processing
- **Alignment Strategy Comparison**: A/B testing performance
- **Memory Usage Scaling**: Memory consumption validation
- **Processing Throughput**: Segments per second calculation
- **Service Bottleneck Identification**: Performance profiling
- **Error Handling Performance**: Failure response time

**Slow Performance Tests** (`@pytest.mark.slow`):
- **Resource Cleanup Performance**: Multi-iteration leak detection
- **Scalability Limits**: Large dataset processing (100-1000 segments)

#### Realistic Performance Expectations

Tests calculate expected times based on mock delays:

```python
expected_time = (
    CONFIG["data_extraction_delay"] +
    CONFIG["translation_delay_per_text"] * CONFIG["segment_count"] +
    CONFIG["alignment_delay_per_segment"] * CONFIG["segment_count"] +
    CONFIG["tts_delay_per_audio"] * CONFIG["segment_count"]
)
max_time = expected_time * 1.5  # 50% buffer for overhead
```

#### Execution Commands

```bash
# Run all performance tests
make test-performance

# Run only fast performance tests (CI-friendly)
make test-performance-fast

# Run only slow performance tests (comprehensive validation)
make test-performance-slow

# Environment-specific execution
CI=true make test-performance-fast    # Force CI mode
LOCAL=true make test-performance-slow # Force local mode
```

### 4. Evaluation Tests (`evaluation/`)

**Purpose**: Quality assessment and human-readable validation of pipeline steps.

**Characteristics**:
- Real service integration (when API keys available)
- Human-readable before/after comparisons
- Configuration A/B testing
- Quality scoring and metrics

**Current Implementation**:
- **Transcript Processing Evaluation**: Text enhancement and processing quality
- **Alignment Evaluation**: Timing synchronization accuracy

**Execution**:
```bash
# Run evaluation framework
poetry run python3 evaluation/scripts/run_evaluation.py --help

# Specific step evaluation
poetry run python3 evaluation/scripts/run_evaluation.py --step transcript_processing
```

**Integration with Testing**:
- Performance tests use evaluation metrics for quality validation
- A/B testing framework validates alignment strategies
- Quality scores feed into performance benchmarks

## Test Markers and Organization

### Pytest Markers

```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.performance   # Performance tests
@pytest.mark.fast          # Fast-running tests (<5s)
@pytest.mark.slow          # Slow-running tests (>10s)
@pytest.mark.requires_api  # Tests requiring external API keys
```

### Marker Usage Examples

```python
@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.fast
async def test_pipeline_processing_time(self, performance_pipeline):
    """Fast performance test suitable for CI."""
    
@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
async def test_scalability_limits(self, performance_mock_services, tmp_path):
    """Comprehensive scalability test for local validation."""
```

### Test Selection

```bash
# Run only fast tests
pytest -m fast

# Run only performance tests
pytest -m performance

# Run integration tests excluding slow ones
pytest -m "integration and not slow"

# Run API tests (if keys available)
pytest -m requires_api
```

## Mock Strategy

### Service Mocking Principles

1. **Realistic Behavior**: Mocks simulate real service characteristics
2. **Configurable Delays**: Environment-appropriate timing simulation
3. **Error Simulation**: Test error handling paths
4. **Data Validation**: Ensure mocks use correct data models

### Mock Implementation Patterns

#### Service Mocks with Realistic Delays

```python
async def mock_translate_batch(texts, target_language):
    await asyncio.sleep(CONFIG["translation_delay_per_text"] * len(texts))
    return [f"Translated: {text}" for text in texts]

translation_service.translate_batch = mock_translate_batch
```

#### Error Injection for Testing

```python
# Test error handling
service.translate_batch = AsyncMock(side_effect=Exception("Translation error"))
```

#### Environment-Aware Configuration

```python
@pytest.fixture
def performance_mock_services(self, tmp_path, large_timed_transcript):
    """Create mock services with environment-appropriate delays."""
    # Services automatically use CONFIG delays based on environment
```

## CI/CD Integration

### GitHub Actions Configuration

**Fast Test Suite** (every PR):
```yaml
- name: Run Fast Tests
  run: |
    make test-fast
    make test-performance-fast
```

**Comprehensive Test Suite** (main branch):
```yaml
- name: Run All Tests
  run: |
    make test
    make test-performance
```

### Environment Variables

- `CI=true` or `GITHUB_ACTIONS=true`: Triggers CI mode with reduced test scope
- `OPENAI_API_KEY`: Enables API-dependent tests
- `PYTEST_TIMEOUT`: Override default test timeouts

## Best Practices

### Test Development Guidelines

1. **Start with Unit Tests**: Test components in isolation first
2. **Realistic Integration**: Use realistic data sizes and timing
3. **Environment Awareness**: Consider both CI and local development needs
4. **Performance Baselines**: Establish and maintain performance benchmarks
5. **Error Coverage**: Test both success and failure scenarios

### Performance Test Design

1. **Calculate Expected Times**: Base assertions on mock delays, not arbitrary values
2. **Environment Scaling**: Use different parameters for CI vs local testing
3. **Buffer for Overhead**: Allow 50% buffer for test framework overhead
4. **Meaningful Metrics**: Track throughput, memory usage, and bottlenecks

### Mock Design Principles

1. **Behavior Over Data**: Focus on realistic behavior patterns
2. **Configurable Timing**: Make delays environment-aware
3. **Error Scenarios**: Include failure modes in mocks
4. **Interface Compliance**: Ensure mocks match real service interfaces

## Troubleshooting

### Common Issues

**Test Timeouts**:
- Check environment configuration (CI vs LOCAL)
- Verify mock delays are appropriate
- Ensure realistic performance expectations

**Memory Usage**:
- Monitor memory consumption in performance tests
- Use `psutil` for memory tracking
- Set appropriate memory limits

**API Rate Limits**:
- Use `@pytest.mark.requires_api` for external service tests
- Implement proper retry logic
- Mock external services for most tests

### Debugging Performance Tests

```bash
# Run with verbose output
pytest tests/integration/test_performance_integration.py -v -s

# Run specific test with timing details
pytest tests/integration/test_performance_integration.py::TestPerformanceIntegration::test_pipeline_processing_time -v -s

# Force environment mode
CI=true pytest tests/integration/test_performance_integration.py -v
```

## Future Considerations

### Planned Enhancements

1. **Property-Based Testing**: Add hypothesis-based testing for edge cases
2. **Load Testing**: Implement realistic load testing scenarios
3. **Chaos Testing**: Add failure injection and recovery validation
4. **Regression Testing**: Automated performance regression detection

### Scaling Strategy

As the project grows:
- Add more granular test categories
- Implement test parallelization
- Enhance environment-specific configurations
- Add more sophisticated performance benchmarking

## References

- **ADR-0004**: Pipeline Step Evaluation Framework
- **ADR-0005**: Pipeline Step Separation and Responsibility Alignment
- **Evaluation Framework**: `evaluation/README.md`
- **Performance Analysis**: `docs/performance_analysis_phase4.md`
- **Developer Onboarding**: `onboarding/developer_onboarding.json`