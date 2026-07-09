import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from .evaluate import EvaluationResult
from training.utils import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: str = "./eval_reports") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_markdown(
        self,
        results: Dict[str, EvaluationResult],
        model_name: str = "unknown",
        title: Optional[str] = None,
    ) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        report_title = title or f"Evaluation Report — {model_name}"

        lines = [
            f"# {report_title}",
            f"",
            f"**Model:** {model_name}  ",
            f"**Date:** {timestamp}  ",
            f"**Tests:** {len(results)}  ",
            f"",
            "---",
            "",
            "## Summary",
            "",
            "| Benchmark | Accuracy | Avg Latency (ms) | Throughput (tok/s) | Peak Memory (MB) | Samples |",
            "|-----------|----------|------------------|---------------------|-------------------|---------|",
        ]

        for name, result in results.items():
            acc = f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
            lines.append(
                f"| {name} | {acc} | {result.avg_latency_ms:.1f} | "
                f"{result.throughput_tokens_per_sec:.1f} | {result.peak_memory_mb:.1f} | "
                f"{result.num_samples} |"
            )

        lines.extend(["", "## Per-Benchmark Details", ""])

        for name, result in results.items():
            lines.extend([
                f"### {name}",
                f"",
                f"- **Accuracy:** {result.accuracy:.4f}" if result.accuracy is not None else "- **Accuracy:** N/A",
                f"- **Average Latency:** {result.avg_latency_ms:.2f} ms",
                f"- **P50 Latency:** {result.p50_latency_ms:.2f} ms",
                f"- **P95 Latency:** {result.p95_latency_ms:.2f} ms",
                f"- **P99 Latency:** {result.p99_latency_ms:.2f} ms",
                f"- **Throughput:** {result.throughput_tokens_per_sec:.1f} tokens/sec",
                f"- **Peak Memory:** {result.peak_memory_mb:.1f} MB",
                f"- **Total Tokens:** {result.total_tokens}",
                f"- **Samples:** {result.num_samples}",
                f"- **Correct:** {result.num_correct}",
                f"- **Errors:** {result.num_errors}",
                f"",
            ])

            if result.per_sample:
                lines.extend(["| # | Correct | Latency (ms) | Tokens |", "|---|---------|--------------|--------|"])
                for idx, sample in enumerate(result.per_sample):
                    lines.append(
                        f"| {idx} | {sample.get('correct', 'N/A')} | "
                        f"{sample.get('latency_ms', 0):.1f} | "
                        f"{sample.get('num_tokens', 0)} |"
                    )
                lines.append("")

        report = "\n".join(lines)
        return report

    def save_markdown(
        self,
        results: Dict[str, EvaluationResult],
        model_name: str = "unknown",
        filename: Optional[str] = None,
    ) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"report_{model_name.replace('/', '_')}_{ts}.md"
        path = os.path.join(self.output_dir, fname)
        report = self.generate_markdown(results, model_name)
        with open(path, "w") as f:
            f.write(report)
        logger.info("Markdown report saved: %s", path)
        return path

    def save_csv(
        self,
        results: Dict[str, EvaluationResult],
        model_name: str = "unknown",
        filename: Optional[str] = None,
    ) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"report_{model_name.replace('/', '_')}_{ts}.csv"
        path = os.path.join(self.output_dir, fname)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Benchmark", "Accuracy", "AvgLatency_ms", "P50Latency_ms",
                "P95Latency_ms", "P99Latency_ms", "Throughput_tok_s",
                "PeakMemory_MB", "TotalTokens", "Samples", "Correct", "Errors",
            ])
            for name, result in results.items():
                writer.writerow([
                    name,
                    f"{result.accuracy:.6f}" if result.accuracy is not None else "",
                    f"{result.avg_latency_ms:.2f}",
                    f"{result.p50_latency_ms:.2f}",
                    f"{result.p95_latency_ms:.2f}",
                    f"{result.p99_latency_ms:.2f}",
                    f"{result.throughput_tokens_per_sec:.2f}",
                    f"{result.peak_memory_mb:.2f}",
                    result.total_tokens,
                    result.num_samples,
                    result.num_correct,
                    result.num_errors,
                ])
        logger.info("CSV report saved: %s", path)
        return path

    def save_html(
        self,
        results: Dict[str, EvaluationResult],
        model_name: str = "unknown",
        filename: Optional[str] = None,
    ) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = filename or f"report_{model_name.replace('/', '_')}_{ts}.html"
        path = os.path.join(self.output_dir, fname)

        rows = ""
        for name, result in results.items():
            acc = f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A"
            rows += f"""
            <tr>
                <td>{name}</td>
                <td>{acc}</td>
                <td>{result.avg_latency_ms:.1f}</td>
                <td>{result.p50_latency_ms:.1f}</td>
                <td>{result.p95_latency_ms:.1f}</td>
                <td>{result.p99_latency_ms:.1f}</td>
                <td>{result.throughput_tokens_per_sec:.1f}</td>
                <td>{result.peak_memory_mb:.1f}</td>
                <td>{result.total_tokens}</td>
                <td>{result.num_samples}</td>
                <td>{result.num_correct}</td>
                <td>{result.num_errors}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluation Report — {model_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #f5f5f5; }}
h1 {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #e0e0e0; }}
th {{ background: #4a90d9; color: white; }}
tr:hover {{ background: #f0f6ff; }}
.meta {{ color: #666; margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<div class="meta">
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Date:</strong> {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
    <p><strong>Benchmarks:</strong> {len(results)}</p>
</div>
<table>
<thead>
<tr>
    <th>Benchmark</th><th>Accuracy</th><th>Avg Lat (ms)</th><th>P50 (ms)</th>
    <th>P95 (ms)</th><th>P99 (ms)</th><th>Throughput (tok/s)</th>
    <th>Peak Mem (MB)</th><th>Tokens</th><th>Samples</th><th>Correct</th><th>Errors</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)
        logger.info("HTML report saved: %s", path)
        return path

    def save_all(
        self,
        results: Dict[str, EvaluationResult],
        model_name: str = "unknown",
    ) -> Dict[str, str]:
        return {
            "markdown": self.save_markdown(results, model_name),
            "csv": self.save_csv(results, model_name),
            "html": self.save_html(results, model_name),
        }
