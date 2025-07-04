"""Base classes for pipeline step evaluation framework."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class EvaluationDataset:
    """Curated dataset for pipeline step evaluation."""
    name: str
    description: str
    input_samples: List[Any]
    expected_outcomes: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "EvaluationDataset":
        """Load evaluation dataset from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save evaluation dataset to JSON file."""
        with open(file_path, 'w') as f:
            json.dump({
                'name': self.name,
                'description': self.description,
                'input_samples': self.input_samples,
                'expected_outcomes': self.expected_outcomes,
                'metadata': self.metadata
            }, f, indent=2)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    processing_mode: Optional[str] = None
    max_samples: Optional[int] = None
    include_timing: bool = True
    include_quality_metrics: bool = True
    config_name: str = "default"
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Results from evaluating a pipeline step."""
    step_name: str
    dataset_name: str
    config_name: str
    input_data: Any
    output_data: Any
    metrics: Dict[str, float]
    quality_score: float
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    error_info: Optional[str] = None
    human_annotations: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'step_name': self.step_name,
            'dataset_name': self.dataset_name,
            'config_name': self.config_name,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'metrics': self.metrics,
            'quality_score': self.quality_score,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp.isoformat(),
            'error_info': self.error_info,
            'human_annotations': self.human_annotations
        }


@dataclass
class ComparisonReport:
    """Report comparing different configurations."""
    configurations: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    quality_comparison: Dict[str, float]
    performance_comparison: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class PipelineStepEvaluator(ABC):
    """Abstract base class for pipeline step evaluators."""
    
    def __init__(self, step_name: str):
        """Initialize evaluator for a specific pipeline step."""
        self.step_name = step_name
        self.logger = logging.getLogger(f"{__name__}.{step_name}")
    
    @abstractmethod
    async def evaluate_step(
        self, 
        dataset: EvaluationDataset, 
        config: EvaluationConfig
    ) -> List[EvaluationResult]:
        """Run evaluation on a pipeline step with given dataset."""
        pass
    
    @abstractmethod
    def calculate_quality_score(self, input_data: Any, output_data: Any) -> float:
        """Calculate quality score for a single evaluation."""
        pass
    
    @abstractmethod
    def extract_metrics(self, input_data: Any, output_data: Any) -> Dict[str, float]:
        """Extract step-specific metrics from input/output."""
        pass
    
    def generate_report(
        self, 
        results: List[EvaluationResult], 
        format: ReportFormat = ReportFormat.MARKDOWN
    ) -> str:
        """Generate human-readable evaluation report."""
        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(results)
        elif format == ReportFormat.HTML:
            return self._generate_html_report(results)
        elif format == ReportFormat.JSON:
            return self._generate_json_report(results)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def compare_configurations(
        self, 
        results_by_config: Dict[str, List[EvaluationResult]]
    ) -> ComparisonReport:
        """Compare different processing configurations."""
        configurations = list(results_by_config.keys())
        metrics_comparison = {}
        quality_comparison = {}
        performance_comparison = {}
        
        # Calculate average metrics for each configuration
        for config_name, results in results_by_config.items():
            # Average quality score
            quality_scores = [r.quality_score for r in results if r.error_info is None]
            quality_comparison[config_name] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            # Average processing time
            processing_times = [r.processing_time for r in results if r.error_info is None]
            performance_comparison[config_name] = sum(processing_times) / len(processing_times) if processing_times else 0.0
            
            # Average metrics
            all_metrics = {}
            for result in results:
                if result.error_info is None:
                    for metric_name, value in result.metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)
            
            metrics_comparison[config_name] = {
                metric: sum(values) / len(values) if values else 0.0
                for metric, values in all_metrics.items()
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            configurations, quality_comparison, performance_comparison, metrics_comparison
        )
        
        return ComparisonReport(
            configurations=configurations,
            metrics_comparison=metrics_comparison,
            quality_comparison=quality_comparison,
            performance_comparison=performance_comparison,
            recommendations=recommendations
        )
    
    def _generate_markdown_report(self, results: List[EvaluationResult]) -> str:
        """Generate markdown evaluation report."""
        successful_results = [r for r in results if r.error_info is None]
        failed_results = [r for r in results if r.error_info is not None]
        
        report = f"""# {self.step_name} Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Samples:** {len(results)}
- **Successful:** {len(successful_results)}
- **Failed:** {len(failed_results)}
- **Success Rate:** {len(successful_results) / len(results) * 100:.1f}%

"""
        
        if successful_results:
            avg_quality = sum(r.quality_score for r in successful_results) / len(successful_results)
            avg_time = sum(r.processing_time for r in successful_results) / len(successful_results)
            
            report += f"""## Performance Metrics

- **Average Quality Score:** {avg_quality:.3f}
- **Average Processing Time:** {avg_time:.3f}s

"""
            
            # Add visual comparison section
            report += self._generate_visual_comparison_section(successful_results[:5])
            
            report += f"""## Technical Metrics

"""
            
            for i, result in enumerate(successful_results[:5]):  # Show first 5 results
                report += f"""### Sample {i+1} - Technical Details

**Quality Score:** {result.quality_score:.3f} | **Processing Time:** {result.processing_time:.3f}s

**Metrics:**
"""
                for metric, value in result.metrics.items():
                    report += f"- {metric}: {value:.3f}\n"
                
                report += "\n"
        
        if failed_results:
            report += f"""## Failed Samples ({len(failed_results)})

"""
            for i, result in enumerate(failed_results):
                report += f"""### Failed Sample {i+1}

**Error:** {result.error_info}

**Input:** {str(result.input_data)[:200]}...

"""
        
        return report
    
    def _generate_visual_comparison_section(self, results: List[EvaluationResult]) -> str:
        """Generate visual before/after comparison section."""
        if not results:
            return ""
            
        section = "## Input vs Output Comparison\n\n"
        
        for i, result in enumerate(results):
            section += f"### Sample {i+1}\n\n"
            
            # Get description if available
            description = ""
            if isinstance(result.input_data, dict):
                description = result.input_data.get("description", "")
            if description:
                section += f"**Description:** {description}\n\n"
            
            # Generate step-specific visual comparison
            step_comparison = self._generate_step_specific_comparison(result.input_data, result.output_data)
            section += step_comparison
            
            section += "\n---\n\n"
        
        return section
    
    def _generate_step_specific_comparison(self, input_data: Any, output_data: Any) -> str:
        """Generate step-specific visual comparison. Override in subclasses."""
        return f"""**Before:**
```
{str(input_data)[:300]}...
```

**After:**
```
{str(output_data)[:300]}...
```

"""
    
    def _generate_html_report(self, results: List[EvaluationResult]) -> str:
        """Generate HTML evaluation report."""
        # Basic HTML report - can be enhanced with templates
        markdown_report = self._generate_markdown_report(results)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.step_name} Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .summary {{ background-color: #f5f5f5; padding: 20px; margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .error {{ color: red; }}
        pre {{ background-color: #f0f0f0; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <pre>{markdown_report}</pre>
</body>
</html>"""
        return html
    
    def _generate_json_report(self, results: List[EvaluationResult]) -> str:
        """Generate JSON evaluation report."""
        report_data = {
            'step_name': self.step_name,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_samples': len(results),
                'successful': len([r for r in results if r.error_info is None]),
                'failed': len([r for r in results if r.error_info is not None]),
            },
            'results': [result.to_dict() for result in results]
        }
        return json.dumps(report_data, indent=2)
    
    def _generate_recommendations(
        self, 
        configurations: List[str], 
        quality_comparison: Dict[str, float],
        performance_comparison: Dict[str, float],
        metrics_comparison: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Find best quality configuration
        best_quality_config = max(quality_comparison.items(), key=lambda x: x[1])
        recommendations.append(
            f"**Best Quality:** {best_quality_config[0]} (score: {best_quality_config[1]:.3f})"
        )
        
        # Find fastest configuration
        fastest_config = min(performance_comparison.items(), key=lambda x: x[1])
        recommendations.append(
            f"**Fastest:** {fastest_config[0]} (time: {fastest_config[1]:.3f}s)"
        )
        
        # Quality vs performance trade-off
        quality_scores = list(quality_comparison.values())
        performance_scores = list(performance_comparison.values())
        
        if len(quality_scores) > 1:
            quality_range = max(quality_scores) - min(quality_scores)
            performance_range = max(performance_scores) - min(performance_scores)
            
            if quality_range > 0.1:  # Significant quality difference
                recommendations.append(
                    "**Quality varies significantly between configurations - consider use case requirements**"
                )
            
            if performance_range > 1.0:  # Significant performance difference
                recommendations.append(
                    "**Performance varies significantly - consider throughput requirements**"
                )
        
        return recommendations


async def run_evaluation_sample(
    input_data: Any, 
    evaluator: PipelineStepEvaluator,
    config: EvaluationConfig
) -> EvaluationResult:
    """Run evaluation on a single sample with timing and error handling."""
    start_time = time.time()
    
    try:
        # This will be implemented by specific evaluators
        output_data = await evaluator._process_sample(input_data, config)
        processing_time = time.time() - start_time
        
        # Calculate metrics and quality score
        metrics = evaluator.extract_metrics(input_data, output_data)
        quality_score = evaluator.calculate_quality_score(input_data, output_data)
        
        return EvaluationResult(
            step_name=evaluator.step_name,
            dataset_name="",  # Will be filled by caller
            config_name=config.config_name,
            input_data=input_data,
            output_data=output_data,
            metrics=metrics,
            quality_score=quality_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing sample: {e}")
        
        return EvaluationResult(
            step_name=evaluator.step_name,
            dataset_name="",  # Will be filled by caller
            config_name=config.config_name,
            input_data=input_data,
            output_data=None,
            metrics={},
            quality_score=0.0,
            processing_time=processing_time,
            error_info=str(e)
        )