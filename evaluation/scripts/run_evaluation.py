#!/usr/bin/env python3
"""
Pipeline Step Evaluation Runner

This script runs evaluation datasets through pipeline steps and generates
human-readable reports for quality assessment.

Usage:
    python evaluation/scripts/run_evaluation.py --step transcript_processing --dataset fragmented_speech
    python evaluation/scripts/run_evaluation.py --step transcript_processing --all-datasets
    python evaluation/scripts/run_evaluation.py --step transcript_processing --compare-configs
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from evaluation.evaluators import (
    EvaluationConfig,
    EvaluationDataset,
    ReportFormat,
    TranscriptProcessingEvaluator,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_dataset_path(step: str, dataset_name: str) -> Path:
    """Get path to dataset file."""
    base_path = Path(__file__).parent.parent
    return base_path / "datasets" / step / f"{dataset_name}.json"


def get_report_path(step: str, dataset_name: str, config_name: str = "default") -> Path:
    """Get path for generated report."""
    base_path = Path(__file__).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{step}_{dataset_name}_{config_name}_{timestamp}.md"
    return base_path / "reports" / "generated" / report_name


def load_dataset(step: str, dataset_name: str) -> EvaluationDataset:
    """Load evaluation dataset from file."""
    dataset_path = get_dataset_path(step, dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    return EvaluationDataset.from_json(dataset_path)


def get_evaluator(step: str):
    """Get evaluator for pipeline step."""
    if step == "transcript_processing":
        return TranscriptProcessingEvaluator()
    else:
        raise ValueError(f"Unknown step: {step}")


def list_available_datasets(step: str) -> List[str]:
    """List available datasets for a step."""
    datasets_path = Path(__file__).parent.parent / "datasets" / step
    if not datasets_path.exists():
        return []
    
    return [
        f.stem for f in datasets_path.glob("*.json")
        if f.is_file()
    ]


async def run_single_evaluation(
    step: str,
    dataset_name: str,
    config: EvaluationConfig,
    output_format: ReportFormat = ReportFormat.MARKDOWN
) -> str:
    """Run evaluation on a single dataset with given configuration."""
    logger.info(f"Running evaluation: {step} / {dataset_name} / {config.config_name}")
    
    # Load dataset
    dataset = load_dataset(step, dataset_name)
    logger.info(f"Loaded dataset: {dataset.name} ({len(dataset.input_samples)} samples)")
    
    # Get evaluator
    evaluator = get_evaluator(step)
    
    # Run evaluation
    results = await evaluator.evaluate_step(dataset, config)
    
    # Generate report
    report = evaluator.generate_report(results, output_format)
    
    # Save report
    report_path = get_report_path(step, dataset_name, config.config_name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    
    return str(report_path)


async def run_all_datasets(
    step: str,
    config: EvaluationConfig,
    output_format: ReportFormat = ReportFormat.MARKDOWN
) -> List[str]:
    """Run evaluation on all available datasets for a step."""
    datasets = list_available_datasets(step)
    if not datasets:
        logger.warning(f"No datasets found for step: {step}")
        return []
    
    logger.info(f"Running evaluation on {len(datasets)} datasets: {', '.join(datasets)}")
    
    report_paths = []
    for dataset_name in datasets:
        try:
            report_path = await run_single_evaluation(step, dataset_name, config, output_format)
            report_paths.append(report_path)
        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            continue
    
    return report_paths


async def compare_configurations(
    step: str,
    dataset_name: str,
    configs: List[EvaluationConfig],
    output_format: ReportFormat = ReportFormat.MARKDOWN
) -> str:
    """Compare different configurations on the same dataset."""
    logger.info(f"Comparing configurations for {step} / {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(step, dataset_name)
    
    # Get evaluator
    evaluator = get_evaluator(step)
    
    # Run evaluation for each configuration
    results_by_config = {}
    for config in configs:
        logger.info(f"Evaluating configuration: {config.config_name}")
        try:
            results = await evaluator.evaluate_step(dataset, config)
            results_by_config[config.config_name] = results
        except Exception as e:
            logger.error(f"Error evaluating config {config.config_name}: {e}")
            continue
    
    if not results_by_config:
        raise RuntimeError("No successful evaluations for comparison")
    
    # Generate comparison report
    comparison = evaluator.compare_configurations(results_by_config)
    
    # Create comparison report
    report = f"""# Configuration Comparison Report

**Step:** {step}  
**Dataset:** {dataset_name}  
**Generated:** {comparison.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Configurations Compared

{', '.join(comparison.configurations)}

## Quality Comparison

"""
    
    for config, score in comparison.quality_comparison.items():
        report += f"- **{config}**: {score:.3f}\n"
    
    report += "\n## Performance Comparison\n\n"
    
    for config, time in comparison.performance_comparison.items():
        report += f"- **{config}**: {time:.3f}s\n"
    
    report += "\n## Metrics Comparison\n\n"
    
    # Show metrics table
    all_metrics = set()
    for config_metrics in comparison.metrics_comparison.values():
        all_metrics.update(config_metrics.keys())
    
    if all_metrics:
        report += "| Metric | " + " | ".join(comparison.configurations) + " |\n"
        report += "|--------|" + "|".join(["--------"] * len(comparison.configurations)) + "|\n"
        
        for metric in sorted(all_metrics):
            row = f"| {metric} | "
            for config in comparison.configurations:
                value = comparison.metrics_comparison.get(config, {}).get(metric, 0.0)
                row += f"{value:.3f} | "
            report += row + "\n"
    
    report += "\n## Recommendations\n\n"
    
    for rec in comparison.recommendations:
        report += f"- {rec}\n"
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(__file__).parent.parent / "reports" / "generated" / f"{step}_{dataset_name}_comparison_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Comparison report saved to: {report_path}")
    
    return str(report_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pipeline step evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single evaluation
  python evaluation/scripts/run_evaluation.py --step transcript_processing --dataset fragmented_speech
  
  # Run all datasets for a step
  python evaluation/scripts/run_evaluation.py --step transcript_processing --all-datasets
  
  # Compare configurations
  python evaluation/scripts/run_evaluation.py --step transcript_processing --dataset fragmented_speech --compare-configs
  
  # List available datasets
  python evaluation/scripts/run_evaluation.py --step transcript_processing --list-datasets
        """
    )
    
    parser.add_argument(
        "--step",
        required=True,
        choices=["transcript_processing"],
        help="Pipeline step to evaluate"
    )
    
    parser.add_argument(
        "--dataset",
        help="Dataset name to evaluate"
    )
    
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run evaluation on all available datasets"
    )
    
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Compare different configurations"
    )
    
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets for the step"
    )
    
    parser.add_argument(
        "--config",
        default="default",
        help="Configuration name"
    )
    
    parser.add_argument(
        "--processing-mode",
        choices=["rule_based", "ai_enhanced", "hybrid"],
        default="hybrid",
        help="Processing mode for transcript processing"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Output report format"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List datasets
    if args.list_datasets:
        datasets = list_available_datasets(args.step)
        if datasets:
            print(f"Available datasets for {args.step}:")
            for dataset in datasets:
                print(f"  - {dataset}")
        else:
            print(f"No datasets found for {args.step}")
        return
    
    # Create configuration
    config = EvaluationConfig(
        config_name=args.config,
        processing_mode=args.processing_mode,
        max_samples=args.max_samples
    )
    
    # Determine output format
    format_map = {
        "markdown": ReportFormat.MARKDOWN,
        "html": ReportFormat.HTML,
        "json": ReportFormat.JSON
    }
    output_format = format_map[args.output_format]
    
    # Run evaluation
    try:
        if args.compare_configs:
            if not args.dataset:
                parser.error("--dataset is required for --compare-configs")
            
            # Create configurations to compare
            configs = [
                EvaluationConfig(config_name="rule_based", processing_mode="rule_based"),
                EvaluationConfig(config_name="ai_enhanced", processing_mode="ai_enhanced"),
                EvaluationConfig(config_name="hybrid", processing_mode="hybrid")
            ]
            
            report_path = asyncio.run(
                compare_configurations(args.step, args.dataset, configs, output_format)
            )
            print(f"Comparison report generated: {report_path}")
            
        elif args.all_datasets:
            report_paths = asyncio.run(
                run_all_datasets(args.step, config, output_format)
            )
            print(f"Generated {len(report_paths)} reports:")
            for path in report_paths:
                print(f"  - {path}")
                
        elif args.dataset:
            report_path = asyncio.run(
                run_single_evaluation(args.step, args.dataset, config, output_format)
            )
            print(f"Report generated: {report_path}")
            
        else:
            parser.error("Must specify --dataset, --all-datasets, or --compare-configs")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()