#!/usr/bin/env python3
"""
Merge two evaluation results directories into one.

This module checks if two results directories have compatible configurations
(excluding sample-selection fields like max_samples, sample_indices, filter_numerical),
and merges them into a single output directory with combined statistics.

Usage:
    python evaluate/merge_results.py results/run_A results/run_B --output results/merged
    
Example:
    python evaluate/merge_results.py \
        evaluate/results/ExCiAn_cmprs_3_rounds \
        evaluate/results/ExCiAn_cmprs_5_rounds \
        --output evaluate/results/merged_results
"""

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Fields to ignore when comparing configurations (sample-selection fields)
IGNORE_CONFIG_FIELDS = {
    'max_samples',
    'sample_indices', 
    'filter_numerical',
    'sample_range',
}


def deep_compare_configs(config1: Any, config2: Any, path: str = "", ignore_fields: set = None) -> List[str]:
    """
    Recursively compare two config structures, returning list of differences.
    
    Args:
        config1: First config value
        config2: Second config value  
        path: Current path in config (for error messages)
        ignore_fields: Set of field names to ignore
        
    Returns:
        List of difference descriptions (empty if configs match)
    """
    if ignore_fields is None:
        ignore_fields = IGNORE_CONFIG_FIELDS
        
    differences = []
    
    if isinstance(config1, dict) and isinstance(config2, dict):
        all_keys = set(config1.keys()) | set(config2.keys())
        for key in all_keys:
            if key in ignore_fields:
                continue
            key_path = f"{path}.{key}" if path else key
            if key not in config1:
                differences.append(f"{key_path}: missing in first config")
            elif key not in config2:
                differences.append(f"{key_path}: missing in second config")
            else:
                differences.extend(deep_compare_configs(
                    config1[key], config2[key], key_path, ignore_fields
                ))
    elif isinstance(config1, list) and isinstance(config2, list):
        if len(config1) != len(config2):
            differences.append(f"{path}: list lengths differ ({len(config1)} vs {len(config2)})")
        else:
            for i, (v1, v2) in enumerate(zip(config1, config2)):
                differences.extend(deep_compare_configs(
                    v1, v2, f"{path}[{i}]", ignore_fields
                ))
    elif config1 != config2:
        differences.append(f"{path}: {config1!r} != {config2!r}")
        
    return differences


def configs_compatible(dir1: Path, dir2: Path) -> Tuple[bool, List[str]]:
    """
    Check if two results directories have compatible configurations.
    
    Args:
        dir1: First results directory
        dir2: Second results directory
        
    Returns:
        Tuple of (is_compatible, list_of_differences)
    """
    config_file = "config_parsed.json"
    
    config1_path = dir1 / config_file
    config2_path = dir2 / config_file
    
    if not config1_path.exists():
        return False, [f"Missing {config_file} in {dir1}"]
    if not config2_path.exists():
        return False, [f"Missing {config_file} in {dir2}"]
    
    try:
        with open(config1_path, 'r') as f:
            config1 = json.load(f)
        with open(config2_path, 'r') as f:
            config2 = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON parse error: {e}"]
    
    differences = deep_compare_configs(config1, config2)
    
    return len(differences) == 0, differences


def merge_results_json(dir1: Path, dir2: Path) -> Tuple[dict, Dict[str, dict]]:
    """
    Merge financebench_results.json from both directories.
    
    Args:
        dir1: First results directory
        dir2: Second results directory
        
    Returns:
        Tuple of (merged_results_dict, sample_id_to_result_map)
    """
    results1_path = dir1 / "financebench_results.json"
    results2_path = dir2 / "financebench_results.json"
    
    with open(results1_path, 'r') as f:
        results1 = json.load(f)
    with open(results2_path, 'r') as f:
        results2 = json.load(f)
    
    # Merge results per protocol
    merged_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "protocols": list(set(results1.get("protocols", []) + results2.get("protocols", []))),
        "total_samples": 0,
        "results": {}
    }
    
    sample_map = {}  # sample_id -> result dict
    
    all_protocols = set(results1.get("results", {}).keys()) | set(results2.get("results", {}).keys())
    
    for protocol in all_protocols:
        protocol_results = []
        seen_sample_ids = set()
        
        # Add results from dir1
        for result in results1.get("results", {}).get(protocol, []):
            sample_id = result["sample_id"]
            if sample_id not in seen_sample_ids:
                protocol_results.append(result)
                seen_sample_ids.add(sample_id)
                sample_map[sample_id] = result
        
        # Add results from dir2 (skip duplicates)
        for result in results2.get("results", {}).get(protocol, []):
            sample_id = result["sample_id"]
            if sample_id not in seen_sample_ids:
                protocol_results.append(result)
                seen_sample_ids.add(sample_id)
                sample_map[sample_id] = result
        
        # Sort by sample_id for consistent ordering
        protocol_results.sort(key=lambda x: x["sample_id"])
        merged_results["results"][protocol] = protocol_results
    
    # Update total samples (use max across protocols)
    total = 0
    for protocol_results in merged_results["results"].values():
        total = max(total, len(protocol_results))
    merged_results["total_samples"] = total
    
    return merged_results, sample_map


def compute_protocol_stats(results: List[dict]) -> dict:
    """Compute statistics for a protocol's results."""
    if not results:
        return {
            "avg_cost": 0.0,
            "total_cost": 0.0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_samples": 0,
            "successful": 0,
            "failed": 0,
            "avg_time": 0.0,
        }
    
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_input = sum(r.get("input_tokens", 0) for r in results)
    total_output = sum(r.get("output_tokens", 0) for r in results)
    total_time = sum(r.get("execution_time", 0) for r in results)
    n = len(results)
    failed = sum(1 for r in results if r.get("error"))
    
    return {
        "avg_cost": total_cost / n,
        "total_cost": total_cost,
        "avg_input_tokens": total_input / n / 1000,  # in thousands
        "avg_output_tokens": total_output / n / 1000,  # in thousands
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_samples": n,
        "successful": n - failed,
        "failed": failed,
        "avg_time": total_time / n,
    }


def generate_csv(merged_results: dict) -> str:
    """Generate CSV content from merged results."""
    lines = ["Protocol,Avg Cost ($),Total Cost ($),Avg Input Tokens (1k),Avg Output Tokens (1k),Total Samples,Successful,Failed,Avg Time (s)"]
    
    for protocol, results in merged_results["results"].items():
        stats = compute_protocol_stats(results)
        lines.append(
            f"{protocol},{stats['avg_cost']:.4f},{stats['total_cost']:.4f},"
            f"{stats['avg_input_tokens']:.2f},{stats['avg_output_tokens']:.2f},"
            f"{stats['total_samples']},{stats['successful']},{stats['failed']},"
            f"{stats['avg_time']:.2f}"
        )
    
    return "\n".join(lines) + "\n"


@dataclass
class CorrectnessEntry:
    """Represents a single correctness evaluation entry."""
    index: int
    is_correct: bool
    sample_id: str
    question: str
    ground_truth: str
    predicted: str
    verdict: str
    confidence: float
    reasoning: str


def parse_correctness_log(log_path: Path) -> Tuple[Dict[str, CorrectnessEntry], str, str]:
    """
    Parse correctness_evaluation.log file.
    
    Returns:
        Tuple of (sample_id_to_entry_map, header_text, model_name)
    """
    if not log_path.exists():
        return {}, "", "gpt-4o"
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    entries = {}
    
    # Extract header info
    model_match = re.search(r"Model: (\S+)", content)
    model_name = model_match.group(1) if model_match else "gpt-4o"
    
    # Parse individual entries
    # Pattern: numbered entry with [✓ CORRECT] or [✗ WRONG]
    entry_pattern = re.compile(
        r'-{80}\n'
        r'(\d+)\. \[(✓ CORRECT|✗ WRONG)\] (\S+)\n'
        r'-{80}\n'
        r'Question: (.*?)\n'
        r'Ground Truth: (.*?)\n'
        r'Predicted: (.*?)\n\n'
        r'Verdict: (CORRECT|WRONG)\n'
        r'Confidence: ([\d.]+)\n'
        r'Reasoning: (.*?)(?=\n\n-{80}|\n\n={80})',
        re.DOTALL
    )
    
    for match in entry_pattern.finditer(content):
        index = int(match.group(1))
        is_correct = match.group(2) == "✓ CORRECT"
        sample_id = match.group(3)
        question = match.group(4).strip()
        ground_truth = match.group(5).strip()
        predicted = match.group(6).strip()
        verdict = match.group(7)
        confidence = float(match.group(8))
        reasoning = match.group(9).strip()
        
        entries[sample_id] = CorrectnessEntry(
            index=index,
            is_correct=is_correct,
            sample_id=sample_id,
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning
        )
    
    return entries, "", model_name


def generate_correctness_log(entries: Dict[str, CorrectnessEntry], model_name: str) -> str:
    """Generate correctness_evaluation.log content from merged entries."""
    lines = [
        "=" * 80,
        "CORRECTNESS EVALUATION DETAILS",
        f"Model: {model_name}",
        "Numerical Tolerance: 10%",
        "=" * 80,
        ""
    ]
    
    # Sort entries by sample_id for consistent ordering
    sorted_entries = sorted(entries.values(), key=lambda e: e.sample_id)
    
    correct_count = 0
    total_count = len(sorted_entries)
    
    for i, entry in enumerate(sorted_entries, 1):
        if entry.is_correct:
            correct_count += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG"
        
        lines.append("-" * 80)
        lines.append(f"{i}. [{status}] {entry.sample_id}")
        lines.append("-" * 80)
        lines.append(f"Question: {entry.question}")
        lines.append(f"Ground Truth: {entry.ground_truth}")
        lines.append(f"Predicted: {entry.predicted}")
        lines.append("")
        lines.append(f"Verdict: {entry.verdict}")
        lines.append(f"Confidence: {entry.confidence:.2f}")
        lines.append(f"Reasoning: {entry.reasoning}")
        lines.append("")
    
    # Summary
    if total_count > 0:
        accuracy = correct_count / total_count * 100
        lines.append("=" * 80)
        lines.append(f"SUMMARY: {correct_count}/{total_count} correct ({accuracy:.2f}%)")
        lines.append("=" * 80)
    
    return "\n".join(lines)


def generate_summary(merged_results: dict, config: dict, correctness_entries: Dict[str, CorrectnessEntry]) -> str:
    """Generate summary.txt content from merged results."""
    lines = []
    
    # Configuration section
    lines.append("Configuration:")
    lines.append("  Dataset:")
    dataset = config.get("dataset", {})
    lines.append(f"    path: {dataset.get('path', 'N/A')}")
    lines.append(f"    filter_numerical: {str(dataset.get('filter_numerical', False)).lower()}")
    lines.append(f"    samples: merged")
    lines.append("")
    
    lines.append("  Models:")
    models = config.get("models", {})
    local = models.get("local", {})
    remote = models.get("remote", {})
    local_name = local.get("name", "N/A")
    local_temp = local.get("temperature", 0)
    local_ctx = local.get("num_ctx", 4096)
    lines.append(f"    local: {local_name} (temp={local_temp}, ctx={local_ctx})")
    remote_name = remote.get("name", "N/A")
    remote_temp = remote.get("temperature", 0)
    lines.append(f"    remote: {remote_name} (temp={remote_temp})")
    lines.append("")
    
    protocols_config = config.get("protocols", {})
    active = protocols_config.get("active", [])
    common = protocols_config.get("common", {})
    lines.append(f"  Protocols: {', '.join(active)}")
    lines.append(f"    max_rounds: {common.get('max_rounds', 'N/A')}")
    lines.append(f"    num_samples_per_task: {common.get('num_samples_per_task', 1)}")
    lines.append("")
    
    minions_config = protocols_config.get("minions", {})
    if minions_config:
        lines.append("  MINIONS Settings:")
        lines.append(f"    num_tasks_per_round: {minions_config.get('num_tasks_per_round', 'N/A')}")
        lines.append(f"    chunk_fn: {minions_config.get('chunk_fn', 'N/A')}")
        lines.append(f"    max_chunk_size: {minions_config.get('max_chunk_size', 'N/A')}")
        lines.append("")
    
    global_config = config.get("global", {})
    lines.append("  Global:")
    lines.append(f"    output_dir: {global_config.get('output_dir', 'N/A')}")
    lines.append(f"    skip_accuracy: {str(global_config.get('skip_accuracy', False)).lower()}")
    lines.append(f"    use_cache: {str(global_config.get('use_cache', True)).lower()}")
    lines.append("")
    
    lines.append("Total runtime: (merged from multiple runs)")
    lines.append("")
    
    # Per-query details
    lines.append("=" * 80)
    lines.append("PER-QUERY DETAILS")
    lines.append("=" * 80)
    lines.append("")
    
    for protocol, results in merged_results["results"].items():
        lines.append(f"{protocol.upper()}:")
        lines.append("-" * 80)
        lines.append(f"{'Sample ID':<45} {'Cost ($)':<12} {'Input Tok':<12} {'Output Tok':<12}")
        lines.append("-" * 80)
        
        for result in results:
            sample_id = result["sample_id"]
            cost = result.get("cost_usd", 0)
            input_tok = result.get("input_tokens", 0)
            output_tok = result.get("output_tokens", 0)
            lines.append(f"{sample_id:<45} {cost:<12.4f} {input_tok:<12} {output_tok:<12}")
        
        lines.append("")
    
    # Evaluation summary
    lines.append("=" * 80)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Protocol':<15} {'Avg Cost ($)':<15} {'Input Tokens (1k)':<18} {'Output Tokens (1k)':<18}")
    lines.append("-" * 80)
    
    for protocol, results in merged_results["results"].items():
        stats = compute_protocol_stats(results)
        lines.append(
            f"{protocol:<15} {stats['avg_cost']:<15.4f} {stats['avg_input_tokens']:<18.2f} "
            f"{stats['avg_output_tokens']:<18.2f}"
        )
    
    lines.append("=" * 80)
    
    # Accuracy
    if correctness_entries:
        correct = sum(1 for e in correctness_entries.values() if e.is_correct)
        total = len(correctness_entries)
        accuracy = correct / total * 100 if total > 0 else 0
        lines.append(f"")
        lines.append(f"Accuracy rate: {accuracy:.2f}% ({correct}/{total} correct, remote verdict with {remote_name})")
    
    return "\n".join(lines)


def merge_directories(dir1: Path, dir2: Path, output_dir: Path, force: bool = False) -> bool:
    """
    Merge two results directories into output directory.
    
    Args:
        dir1: First results directory
        dir2: Second results directory
        output_dir: Output directory for merged results
        force: If True, overwrite output_dir if it exists
        
    Returns:
        True if merge was successful
    """
    # Check directories exist
    if not dir1.exists():
        print(f"Error: Directory does not exist: {dir1}")
        return False
    if not dir2.exists():
        print(f"Error: Directory does not exist: {dir2}")
        return False
    
    # Check configuration compatibility
    print(f"Checking configuration compatibility...")
    compatible, differences = configs_compatible(dir1, dir2)
    if not compatible:
        print(f"Error: Configurations are not compatible:")
        for diff in differences:
            print(f"  - {diff}")
        return False
    print(f"  ✓ Configurations are compatible")
    
    # Create output directory
    if output_dir.exists():
        if not force:
            print(f"Error: Output directory already exists: {output_dir}")
            print(f"  Use --force to overwrite")
            return False
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Copy config files from dir1
    print(f"Copying configuration files...")
    for config_file in ["config_parsed.json", "config.json", "command.txt", "config_used.config"]:
        src = dir1 / config_file
        if src.exists():
            shutil.copy2(src, output_dir / config_file)
            print(f"  ✓ Copied {config_file}")
    
    # Load config for summary generation
    config = {}
    config_path = output_dir / "config_parsed.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Merge sample_logs
    print(f"Merging sample_logs...")
    sample_logs_dir = output_dir / "sample_logs"
    sample_logs_dir.mkdir(exist_ok=True)
    
    sample_log_count = 0
    seen_files = set()
    
    for src_dir in [dir1, dir2]:
        src_logs = src_dir / "sample_logs"
        if src_logs.exists():
            for f in src_logs.iterdir():
                if f.name not in seen_files:
                    shutil.copy2(f, sample_logs_dir / f.name)
                    seen_files.add(f.name)
                    sample_log_count += 1
    
    print(f"  ✓ Copied {sample_log_count} sample log files")
    
    # Merge minions_logs
    print(f"Merging minions_logs...")
    minions_logs_dir = output_dir / "minions_logs"
    minions_logs_dir.mkdir(exist_ok=True)
    
    minions_log_count = 0
    seen_files = set()
    
    for src_dir in [dir1, dir2]:
        src_logs = src_dir / "minions_logs"
        if src_logs.exists():
            for f in src_logs.iterdir():
                if f.name not in seen_files:
                    shutil.copy2(f, minions_logs_dir / f.name)
                    seen_files.add(f.name)
                    minions_log_count += 1
    
    print(f"  ✓ Copied {minions_log_count} minions log files")
    
    # Merge results JSON
    print(f"Merging results JSON...")
    merged_results, sample_map = merge_results_json(dir1, dir2)
    
    with open(output_dir / "financebench_results.json", 'w') as f:
        json.dump(merged_results, f, indent=2)
    print(f"  ✓ Merged {merged_results['total_samples']} samples")
    
    # Generate CSV
    print(f"Generating CSV...")
    csv_content = generate_csv(merged_results)
    with open(output_dir / "financebench_results.csv", 'w') as f:
        f.write(csv_content)
    print(f"  ✓ Generated financebench_results.csv")
    
    # Merge correctness logs
    print(f"Merging correctness evaluation logs...")
    entries1, _, model1 = parse_correctness_log(dir1 / "correctness_evaluation.log")
    entries2, _, model2 = parse_correctness_log(dir2 / "correctness_evaluation.log")
    
    # Merge entries (dir1 takes precedence for duplicates)
    merged_entries = {**entries2, **entries1}
    
    if merged_entries:
        model_name = model1 or model2 or "gpt-4o"
        correctness_content = generate_correctness_log(merged_entries, model_name)
        with open(output_dir / "correctness_evaluation.log", 'w') as f:
            f.write(correctness_content)
        
        correct = sum(1 for e in merged_entries.values() if e.is_correct)
        print(f"  ✓ Merged {len(merged_entries)} correctness entries ({correct} correct)")
    else:
        merged_entries = {}
        print(f"  ✓ No correctness entries to merge")
    
    # Generate summary
    print(f"Generating summary...")
    summary_content = generate_summary(merged_results, config, merged_entries)
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary_content)
    print(f"  ✓ Generated summary.txt")
    
    # Print final statistics
    print(f"\n{'=' * 60}")
    print(f"MERGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {merged_results['total_samples']}")
    
    for protocol, results in merged_results["results"].items():
        stats = compute_protocol_stats(results)
        print(f"\n{protocol.upper()}:")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")
        print(f"  Avg cost: ${stats['avg_cost']:.4f}")
    
    if merged_entries:
        correct = sum(1 for e in merged_entries.values() if e.is_correct)
        total = len(merged_entries)
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge two evaluation results directories into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge two results directories
    python evaluate/merge_results.py results/run_A results/run_B -o results/merged
    
    # Force overwrite existing output
    python evaluate/merge_results.py results/run_A results/run_B -o results/merged --force
    
    # Check compatibility only (dry run)
    python evaluate/merge_results.py results/run_A results/run_B --check-only
        """
    )
    
    parser.add_argument("dir1", type=Path, help="First results directory")
    parser.add_argument("dir2", type=Path, help="Second results directory")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for merged results")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument("--check-only", action="store_true", help="Only check compatibility, don't merge")
    
    args = parser.parse_args()
    
    # Check-only mode
    if args.check_only:
        compatible, differences = configs_compatible(args.dir1, args.dir2)
        if compatible:
            print("✓ Configurations are compatible")
            sys.exit(0)
        else:
            print("✗ Configurations are NOT compatible:")
            for diff in differences:
                print(f"  - {diff}")
            sys.exit(1)
    
    # Require output for merge
    if not args.output:
        parser.error("--output is required for merging (use --check-only to only check compatibility)")
    
    # Perform merge
    success = merge_directories(args.dir1, args.dir2, args.output, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
