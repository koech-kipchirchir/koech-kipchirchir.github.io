"""
Report generation: Markdown, HTML, and CSV report builders.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.base import BenchmarkResult

logger = logging.getLogger("aios.benchmarks.reporter")


class ReportGenerator:
    """Generates benchmark reports in multiple formats."""

    def __init__(self, output_dir: str | Path = "benchmark_reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: list[BenchmarkResult],
        model_name: str = "",
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate all requested report formats. Returns {format: path}."""
        formats = formats or ["md", "html", "csv"]
        paths: dict[str, Path] = {}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        stem = f"benchmark_report_{timestamp}"

        if "md" in formats:
            p = self._output_dir / f"{stem}.md"
            p.write_text(self._build_markdown(results, model_name), encoding="utf-8")
            paths["md"] = p
            logger.info("Markdown report: %s", p)

        if "html" in formats:
            p = self._output_dir / f"{stem}.html"
            p.write_text(self._build_html(results, model_name), encoding="utf-8")
            paths["html"] = p
            logger.info("HTML report: %s", p)

        if "csv" in formats:
            p = self._output_dir / f"{stem}.csv"
            p.write_text(self._build_csv(results, model_name), encoding="utf-8")
            paths["csv"] = p
            logger.info("CSV report: %s", p)

        return paths

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _build_markdown(self, results: list[BenchmarkResult], model_name: str) -> str:
        lines = [
            "# AIOS Benchmark Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Model:** {model_name or 'N/A'}",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Benchmark | Accuracy | Avg Latency | P95 Latency | Tokens/s | Cost | Errors |",
            "|-----------|----------|-------------|-------------|----------|------|--------|",
        ]
        for r in results:
            lines.append(
                f"| {r.name} | {r.accuracy:.2%} | "
                f"{r.avg_latency_s:.3f}s | {r.p95_latency_s:.3f}s | "
                f"{r.token_throughput_sec:.1f} | "
                f"${r.estimated_cost_usd:.4f} | {r.error_count} |"
            )

        lines.extend(["", "---", "", "## Detailed Results", ""])

        for r in results:
            lines.extend(self._detail_md(r))

        lines.append("")
        return "\n".join(lines)

    def _detail_md(self, r: BenchmarkResult) -> list[str]:
        return [
            f"### {r.name}",
            "",
            f"**Description:** {r.description}",
            f"**Version:** {r.version}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Items | {r.total_items} |",
            f"| Correct | {r.correct} |",
            f"| Accuracy | {r.accuracy:.2%} |",
            f"| Total Latency | {r.total_latency_s:.2f}s |",
            f"| Avg Latency | {r.avg_latency_s:.3f}s |",
            f"| P50 Latency | {r.p50_latency_s:.3f}s |",
            f"| P95 Latency | {r.p95_latency_s:.3f}s |",
            f"| P99 Latency | {r.p99_latency_s:.3f}s |",
            f"| Prompt Tokens | {r.total_prompt_tokens} |",
            f"| Completion Tokens | {r.total_completion_tokens} |",
            f"| Token Throughput | {r.token_throughput_sec:.1f} tok/s |",
            f"| Peak Memory | {r.peak_memory_mb:.1f} MB |",
            f"| Avg Memory | {r.avg_memory_mb:.1f} MB |",
            f"| Peak GPU Memory | {r.peak_gpu_memory_mb:.1f} MB |",
            f"| Avg GPU Util | {r.avg_gpu_util_pct:.1f}% |",
            f"| Estimated Cost | ${r.estimated_cost_usd:.4f} |",
            f"| Errors | {r.error_count} ({r.error_rate:.1%}) |",
            "",
        ]

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIOS Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; background: #fafafa; }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3em; }}
  h2 {{ color: #16213e; margin-top: 1.5em; }}
  h3 {{ color: #0f3460; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 0.6em 0.8em; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #1a1a2e; color: #fff; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f8f8; }}
  .metric {{ font-weight: 600; }}
  .good {{ color: #2e7d32; }}
  .warn {{ color: #f57c00; }}
  .bad {{ color: #c62828; }}
  .summary {{ display: flex; gap: 1em; flex-wrap: wrap; margin: 1em 0; }}
  .summary-card {{ background: #fff; border-radius: 8px; padding: 1em 1.5em;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 180px; }}
  .summary-card h4 {{ margin: 0 0 0.3em; color: #666; font-size: 0.85em; text-transform: uppercase; }}
  .summary-card .value {{ font-size: 1.6em; font-weight: 700; color: #1a1a2e; }}
  .footer {{ margin-top: 2em; font-size: 0.85em; color: #888; border-top: 1px solid #e0e0e0; padding-top: 1em; }}
</style>
</head>
<body>
<h1>AIOS Benchmark Report</h1>
<p><strong>Generated:</strong> {generated}</p>
<p><strong>Model:</strong> {model_name}</p>

<div class="summary">
  <div class="summary-card">
    <h4>Benchmarks</h4>
    <div class="value">{bench_count}</div>
  </div>
  <div class="summary-card">
    <h4>Total Items</h4>
    <div class="value">{total_items}</div>
  </div>
  <div class="summary-card">
    <h4>Overall Accuracy</h4>
    <div class="value {overall_acc_class}">{overall_acc_pct}</div>
  </div>
  <div class="summary-card">
    <h4>Total Cost</h4>
    <div class="value">${total_cost}</div>
  </div>
  <div class="summary-card">
    <h4>Total Errors</h4>
    <div class="value">{total_errors}</div>
  </div>
</div>

<h2>Benchmark Comparison</h2>
<table>
<thead><tr>
  <th>Benchmark</th><th>Accuracy</th><th>Avg Latency</th><th>P95 Latency</th>
  <th>Tokens/s</th><th>Cost</th><th>Errors</th>
</tr></thead>
<tbody>
{summary_rows}
</tbody>
</table>

<h2>Detailed Results</h2>
{details}

<div class="footer">AIOS Benchmark Framework</div>
</body>
</html>"""

    def _build_html(self, results: list[BenchmarkResult], model_name: str) -> str:
        summary_rows = ""
        details = ""
        total_items = sum(r.total_items for r in results)
        total_cost = sum(r.estimated_cost_usd for r in results)
        total_errors = sum(r.error_count for r in results)
        overall_acc = sum(r.correct for r in results) / max(total_items, 1)
        overall_acc_pct = f"{overall_acc:.1%}"
        overall_acc_class = "good" if overall_acc >= 0.7 else ("warn" if overall_acc >= 0.4 else "bad")

        for r in results:
            acc_class = "good" if r.accuracy >= 0.7 else ("warn" if r.accuracy >= 0.4 else "bad")
            summary_rows += (
                f"<tr><td>{r.name}</td>"
                f"<td class='{acc_class}'>{r.accuracy:.1%}</td>"
                f"<td>{r.avg_latency_s:.3f}s</td>"
                f"<td>{r.p95_latency_s:.3f}s</td>"
                f"<td>{r.token_throughput_sec:.1f}</td>"
                f"<td>${r.estimated_cost_usd:.4f}</td>"
                f"<td>{r.error_count}</td></tr>\n"
            )
            details += self._detail_html(r)

        return self.HTML_TEMPLATE.format(
            generated=datetime.now(timezone.utc).isoformat(),
            model_name=model_name or "N/A",
            bench_count=len(results),
            total_items=total_items,
            overall_acc_pct=overall_acc_pct,
            overall_acc_class=overall_acc_class,
            total_cost=f"{total_cost:.4f}",
            total_errors=total_errors,
            summary_rows=summary_rows,
            details=details,
        )

    def _detail_html(self, r: BenchmarkResult) -> str:
        rows = ""
        for key, val in [
            ("Total Items", r.total_items),
            ("Correct", r.correct),
            ("Accuracy", f"{r.accuracy:.2%}"),
            ("Total Latency", f"{r.total_latency_s:.2f}s"),
            ("Avg Latency", f"{r.avg_latency_s:.3f}s"),
            ("P50 Latency", f"{r.p50_latency_s:.3f}s"),
            ("P95 Latency", f"{r.p95_latency_s:.3f}s"),
            ("P99 Latency", f"{r.p99_latency_s:.3f}s"),
            ("Prompt Tokens", r.total_prompt_tokens),
            ("Completion Tokens", r.total_completion_tokens),
            ("Token Throughput", f"{r.token_throughput_sec:.1f} tok/s"),
            ("Peak Memory", f"{r.peak_memory_mb:.1f} MB"),
            ("Avg Memory", f"{r.avg_memory_mb:.1f} MB"),
            ("Peak GPU Memory", f"{r.peak_gpu_memory_mb:.1f} MB"),
            ("Avg GPU Util", f"{r.avg_gpu_util_pct:.1f}%"),
            ("Estimated Cost", f"${r.estimated_cost_usd:.4f}"),
            ("Errors", f"{r.error_count} ({r.error_rate:.1%})"),
        ]:
            rows += f"<tr><td>{key}</td><td>{val}</td></tr>\n"

        return (
            f"<h3>{r.name}</h3>\n"
            f"<p><em>{r.description}</em></p>\n"
            f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>\n"
            f"{rows}</tbody></table>\n"
        )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _build_csv(self, results: list[BenchmarkResult], model_name: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Benchmark", "Accuracy", "AvgLatency_s", "P50Latency_s",
                         "P95Latency_s", "P99Latency_s", "TotalLatency_s",
                         "PromptTokens", "CompletionTokens", "TokenThroughput_sec",
                         "PeakMemoryMB", "AvgMemoryMB", "PeakGPUMemoryMB",
                         "AvgGPUUtilPct", "EstimatedCostUSD", "Errors", "ErrorRate",
                         "TotalItems", "Correct"])

        for r in results:
            writer.writerow([
                r.name,
                f"{r.accuracy:.6f}",
                f"{r.avg_latency_s:.6f}",
                f"{r.p50_latency_s:.6f}",
                f"{r.p95_latency_s:.6f}",
                f"{r.p99_latency_s:.6f}",
                f"{r.total_latency_s:.6f}",
                r.total_prompt_tokens,
                r.total_completion_tokens,
                f"{r.token_throughput_sec:.4f}",
                f"{r.peak_memory_mb:.2f}",
                f"{r.avg_memory_mb:.2f}",
                f"{r.peak_gpu_memory_mb:.2f}",
                f"{r.avg_gpu_util_pct:.2f}",
                f"{r.estimated_cost_usd:.6f}",
                r.error_count,
                f"{r.error_rate:.6f}",
                r.total_items,
                r.correct,
            ])

        return output.getvalue()
