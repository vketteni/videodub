#!/usr/bin/env python3
"""
Test the evaluation framework with a simple mock implementation.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluation.evaluators.base import (
    EvaluationConfig,
    EvaluationDataset,
    EvaluationResult,
    PipelineStepEvaluator,
    ReportFormat,
)


class MockTranscriptEvaluator(PipelineStepEvaluator):
    """Mock evaluator for testing."""
    
    def __init__(self):
        super().__init__("mock_transcript_processing")
    
    async def evaluate_step(
        self, 
        dataset: EvaluationDataset, 
        config: EvaluationConfig
    ) -> List[EvaluationResult]:
        """Mock evaluation that processes samples."""
        results = []
        
        for i, sample in enumerate(dataset.input_samples):
            # Mock processing
            output_data = {
                "processed_segments": [
                    {
                        "processed_text": "This is a mock processed segment.",
                        "quality_score": 0.85,
                        "merged_segments": sample.get("segments", [])
                    }
                ]
            }
            
            result = EvaluationResult(
                step_name=self.step_name,
                dataset_name=dataset.name,
                config_name=config.config_name,
                input_data=sample,
                output_data=output_data,
                metrics=self.extract_metrics(sample, output_data),
                quality_score=self.calculate_quality_score(sample, output_data),
                processing_time=0.1  # Mock processing time
            )
            
            results.append(result)
            
        return results
    
    async def _process_sample(self, input_data: Any, config: EvaluationConfig) -> Any:
        """Mock processing."""
        return {"processed": True}
    
    def calculate_quality_score(self, input_data: Any, output_data: Any) -> float:
        """Mock quality score calculation."""
        return 0.85
    
    def extract_metrics(self, input_data: Any, output_data: Any) -> Dict[str, float]:
        """Mock metrics extraction."""
        return {
            "input_segments": len(input_data.get("segments", [])),
            "output_segments": len(output_data.get("processed_segments", [])),
            "quality_score": 0.85
        }


async def test_framework():
    """Test the evaluation framework."""
    print("Testing Pipeline Step Evaluation Framework...")
    
    # Create mock dataset
    dataset = EvaluationDataset(
        name="test_dataset",
        description="Test dataset for framework validation",
        input_samples=[
            {
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "hello world", "speaker": "Test"},
                    {"start": 2.0, "end": 4.0, "text": "this is a test", "speaker": "Test"}
                ]
            },
            {
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": "another test sample", "speaker": "Test"}
                ]
            }
        ]
    )
    
    # Create evaluator
    evaluator = MockTranscriptEvaluator()
    
    # Create config
    config = EvaluationConfig(config_name="test_config")
    
    # Run evaluation
    print("Running evaluation...")
    results = await evaluator.evaluate_step(dataset, config)
    
    print(f"Processed {len(results)} samples")
    for i, result in enumerate(results):
        print(f"  Sample {i+1}: Quality={result.quality_score:.2f}, Time={result.processing_time:.3f}s")
    
    # Generate report
    print("\nGenerating report...")
    report = evaluator.generate_report(results, ReportFormat.MARKDOWN)
    
    # Save report
    report_path = Path("test_evaluation_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    # Test configuration comparison
    print("\nTesting configuration comparison...")
    config2 = EvaluationConfig(config_name="test_config_2")
    results2 = await evaluator.evaluate_step(dataset, config2)
    
    comparison = evaluator.compare_configurations({
        "config_1": results,
        "config_2": results2
    })
    
    print(f"Comparison generated with {len(comparison.recommendations)} recommendations")
    
    print("\n✅ Framework test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_framework())