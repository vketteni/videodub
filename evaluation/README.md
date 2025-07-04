# Pipeline Step Evaluation Framework - Implementation Guide

This document captures critical implementation details and decisions from Phase 1 that are essential for Phase 2 continuation. It supplements ADR-0004 with practical implementation knowledge.

## Phase 1 Implementation Summary

**Status:** ✅ Complete (2025-07-04)  
**ADR Compliance:** 100% + Enhanced human-readable reporting

## Critical Implementation Decisions Not in ADR

### 1. Environment and Dependency Management

**Decision:** Use Poetry environment for evaluation execution
```bash
# CRITICAL: Always run evaluations in Poetry environment
poetry run python3 evaluation/scripts/run_evaluation.py [options]
```

**Why:** The videodub package has dependencies (structlog, pydantic, etc.) that are required for importing real models and services. Running outside Poetry environment falls back to Mock objects.

**Impact for Phase 2:** All new pipeline step evaluators must account for their service dependencies.

### 2. Model Import Strategy

**Challenge:** Direct package imports fail due to relative import dependencies.

**Solution:** Implemented fallback import strategy in `transcript_evaluator.py`:
```python
try:
    # Add src to path and import real models
    from videodub.core.models import ProcessedSegment, ProcessingMode, TranscriptSegment
    # Create mock service that uses real models for testing
    class MockHybridTranscriptProcessingService: ...
except:
    # Complete mock fallback
    from unittest.mock import Mock
```

**Phase 2 Implication:** Each new evaluator needs similar import handling for its respective services.

### 3. Visual Comparison Design Pattern

**Key Enhancement:** Added step-specific visual comparisons beyond the ADR requirements.

**Pattern:**
1. Base class provides generic comparison
2. Each evaluator overrides `_generate_step_specific_comparison()` 
3. Human-readable before/after with change analysis

**Critical for Phase 2:** Translation and TTS evaluators should follow this pattern:
- **Translation:** Show source → target with quality indicators
- **TTS:** Show text → audio properties (duration, quality metrics)

### 4. Dataset Structure Standards

**Established Pattern:**
```json
{
  "name": "dataset_name",
  "description": "Human readable description",
  "input_samples": [...],  // Step-specific input format
  "expected_outcomes": [...],  // Optional reference outputs
  "metadata": {
    "difficulty": "medium",
    "focus_areas": ["area1", "area2"],
    "success_criteria": {...}
  }
}
```

**Critical Fields:**
- `description` field in each sample - used for report section headers
- `metadata.success_criteria` - defines quality thresholds per dataset

### 5. Quality Scoring Philosophy

**Implemented Approach:** Composite scoring (0.0-1.0) combining:
- Functional correctness (does it work?)
- Content preservation (no information loss?)
- Format improvement (capitalization, punctuation, etc.)
- Context coherence (natural flow?)

**Phase 2 Pattern:** Each step should use similar 0.0-1.0 scoring with step-specific components.

### 6. Report Structure Evolution

**Final Structure:**
1. **Summary** - High-level metrics
2. **Performance Metrics** - Averages for quick scanning  
3. **Input vs Output Comparison** - 🔥 **Most valuable section for humans**
4. **Technical Metrics** - Detailed numbers for automation

**Phase 2 Guideline:** Maintain this structure but customize the visual comparison for each step type.

## Phase 2 Prerequisites and Recommendations

### 1. Service Availability Assessment

Before implementing new evaluators, assess:
- **Translation Service:** Can we import and mock it like transcript service?
- **TTS Service:** How do we handle audio output evaluation?
- **Dependencies:** What additional packages might be needed?

### 2. Dataset Creation Strategy

**Learned Pattern:**
- **3 datasets per step minimum:** Different difficulty/scenario types
- **Real-world samples:** Not synthetic - actual problematic cases
- **Progressive difficulty:** Simple → Complex → Edge cases
- **Metadata-driven:** Success criteria defined per dataset

### 3. Evaluation Methodology Expansion

**For Translation Evaluator:**
- Quality metrics: BLEU score, semantic similarity, cultural appropriateness
- Visual comparison: Source → Target with fluency indicators
- Error detection: Mistranslations, cultural misunderstandings

**For TTS Evaluator:**
- Quality metrics: Audio duration matching, naturalness scores
- Visual comparison: Text → Audio properties + sample clips
- Error detection: Mispronunciations, unnatural pauses

### 4. Integration Points Discovered

**Working Integration:**
- ✅ Poetry dependency management
- ✅ Real model imports with fallback
- ✅ CLI tool with multiple options
- ✅ Generated report storage

**Needs Planning for Phase 2:**
- 🔲 CI/CD integration (ADR Phase 3)
- 🔲 Multi-step evaluation workflows
- 🔲 Cross-step quality correlation analysis

## Command Reference

```bash
# List available datasets for a step
poetry run python3 evaluation/scripts/run_evaluation.py --step transcript_processing --list-datasets

# Single evaluation
poetry run python3 evaluation/scripts/run_evaluation.py --step transcript_processing --dataset fragmented_speech

# Compare configurations (A/B testing)
poetry run python3 evaluation/scripts/run_evaluation.py --step transcript_processing --dataset technical_content --compare-configs

# All datasets for a step
poetry run python3 evaluation/scripts/run_evaluation.py --step transcript_processing --all-datasets

# Control sample size and output format
poetry run python3 evaluation/scripts/run_evaluation.py --step transcript_processing --dataset multi_speaker --max-samples 2 --output-format html
```

## File Structure Knowledge

```
evaluation/
├── datasets/
│   └── transcript_processing/          # PATTERN: datasets/{step_name}/{scenario}.json
├── evaluators/
│   ├── base.py                        # Abstract classes - extend for new steps
│   └── transcript_evaluator.py        # TEMPLATE: Copy pattern for new evaluators
├── reports/
│   ├── generated/                     # Auto-generated reports (gitignore these)
│   └── templates/                     # For future HTML template customization
└── scripts/
    ├── run_evaluation.py              # Main CLI - add new steps to choices
    └── test_framework.py              # Development testing tool
```

## Phase 2 Development Checklist

When implementing new pipeline step evaluators:

- [ ] Create `evaluation/datasets/{step_name}/` directory
- [ ] Implement `{Step}Evaluator` class extending `PipelineStepEvaluator`
- [ ] Handle service imports with fallback pattern
- [ ] Override `_generate_step_specific_comparison()` for visual reports
- [ ] Create 3+ curated datasets with real-world scenarios
- [ ] Add step to CLI choices in `run_evaluation.py`
- [ ] Test with `poetry run` to ensure dependencies work
- [ ] Verify reports generate with meaningful visual comparisons

## Key Lessons for Phase 2

1. **Human-readable output is more valuable than metrics** - Focus on visual comparisons first
2. **Real service integration is complex** - Plan import strategy early  
3. **Dataset quality drives evaluation value** - Invest time in realistic scenarios
4. **CLI usability matters** - Make it easy to run common evaluation patterns
5. **Poetry environment is non-negotiable** - All imports must work in managed environment

## Questions for Phase 2 Planning

1. **Audio Handling:** How will TTS evaluation handle audio file comparison and storage?
2. **Translation Quality:** Which translation quality metrics should we implement (BLEU, METEOR, semantic similarity)?
3. **Cross-Step Evaluation:** Should we build workflows that evaluate transcript → translation → TTS as a pipeline?
4. **Performance Benchmarking:** Do we need timing benchmarks for cost/quality trade-off analysis?
5. **Historical Tracking:** Should Phase 2 include evaluation result storage and trend analysis?

---

**This document should be updated as Phase 2 progresses to capture new implementation decisions and patterns.**