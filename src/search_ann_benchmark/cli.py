"""Command-line interface for search-ann-benchmark."""

import os
import sys
from typing import Any

import click

from search_ann_benchmark import __version__
from search_ann_benchmark.config import DATASET_PRESETS, get_dataset_config
from search_ann_benchmark.engines import ENGINE_REGISTRY


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Search ANN Benchmark - Vector search engine performance evaluation suite."""
    pass


@main.command()
@click.argument("engine", type=click.Choice(list(ENGINE_REGISTRY.keys())))
@click.option(
    "-t", "--target",
    default="100k-768-m32-efc200-ef100-ip",
    type=click.Choice(list(DATASET_PRESETS.keys())),
    help="Target dataset configuration",
)
@click.option(
    "-o", "--output",
    default="results.json",
    help="Output file for results",
)
@click.option(
    "--version",
    "engine_version",
    default=None,
    help="Engine version to use",
)
@click.option(
    "--quantization",
    default=None,
    type=click.Choice(["none", "int4", "int8", "bbq", "pq", "byte", "halfvec"]),
    help="Quantization mode",
)
@click.option(
    "--variant",
    default=None,
    help="Product variant identifier",
)
@click.option(
    "--no-filter",
    is_flag=True,
    help="Skip filtered search benchmark",
)
def run(
    engine: str,
    target: str,
    output: str,
    engine_version: str | None,
    quantization: str | None,
    variant: str | None,
    no_filter: bool,
) -> None:
    """Run benchmark for a vector search engine.

    ENGINE is the name of the engine to benchmark (e.g., qdrant, elasticsearch).
    """
    from search_ann_benchmark.engines import get_engine_class
    from search_ann_benchmark.runner import BenchmarkRunner

    # Set environment variables
    if quantization:
        os.environ["SETTING_QUANTIZATION"] = quantization
    if variant:
        os.environ["PRODUCT_VARIANT"] = variant

    click.echo(f"Running benchmark for {engine} with target {target}")

    # Get dataset config
    dataset_config = get_dataset_config(target)
    if quantization:
        dataset_config.quantization = quantization

    # Get engine class and create instance
    engine_class = get_engine_class(engine)

    # Create engine config with optional version override
    engine_config = None
    if engine_version:
        # Get the config class from the engine module
        config_class_name = f"{engine.capitalize()}Config"
        engine_module = sys.modules[engine_class.__module__]
        config_class = getattr(engine_module, config_class_name, None)
        if config_class:
            engine_config = config_class(version=engine_version)

    engine_instance = engine_class(dataset_config, engine_config)

    # Run benchmark
    runner = BenchmarkRunner(engine_instance, target)

    try:
        runner.setup()
        runner.create_index()
        runner.run_indexing()
        runner.run_search_benchmark(page_sizes=[10, 100])

        if not no_filter and runner.section_values:
            runner.run_search_benchmark(page_sizes=[10, 100], with_filter=True)

        runner._print_stats()
        runner.save_results(output)

        click.echo(f"Results saved to {output}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise
    finally:
        runner.cleanup()


@main.command()
def list_engines() -> None:
    """List available vector search engines."""
    click.echo("Available engines:")
    for name in ENGINE_REGISTRY:
        engine_class = ENGINE_REGISTRY[name]
        click.echo(f"  - {name}: {engine_class.__doc__ or 'No description'}")


@main.command()
def list_targets() -> None:
    """List available dataset configurations."""
    click.echo("Available target configurations:")
    for name, config in DATASET_PRESETS.items():
        click.echo(f"  - {name}")
        click.echo(f"      index_size: {config['index_size']:,}")
        click.echo(f"      dimension: {config['dimension']}")
        click.echo(f"      hnsw_m: {config['hnsw_m']}")


@main.command()
@click.argument("engine", type=click.Choice(list(ENGINE_REGISTRY.keys())))
@click.option(
    "-t", "--target",
    default="100k-768-m32-efc200-ef100-ip",
    type=click.Choice(list(DATASET_PRESETS.keys())),
    help="Target dataset configuration",
)
def show_config(engine: str, target: str) -> None:
    """Show configuration for an engine and target."""
    from dataclasses import asdict

    from search_ann_benchmark.engines import get_engine_class

    dataset_config = get_dataset_config(target)
    engine_class = get_engine_class(engine)
    engine_instance = engine_class(dataset_config)

    click.echo(f"Dataset configuration ({target}):")
    for key, value in asdict(dataset_config).items():
        click.echo(f"  {key}: {value}")

    click.echo(f"\nEngine configuration ({engine}):")
    for key, value in engine_instance.engine_config.to_dict().items():
        click.echo(f"  {key}: {value}")


@main.command()
@click.argument("results_file", type=click.Path(exists=True))
def show_results(results_file: str) -> None:
    """Display benchmark results from a JSON file."""
    import json

    with open(results_file) as f:
        results = json.load(f)

    click.echo(f"Benchmark Results: {results.get('target', 'Unknown')}")
    click.echo(f"Engine Version: {results.get('version', 'Unknown')}")
    click.echo(f"Variant: {results.get('variant', 'None')}")
    click.echo(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    click.echo()

    if "indexing" in results.get("results", {}):
        indexing = results["results"]["indexing"]
        click.echo("Indexing:")
        click.echo(f"  Execution time: {indexing.get('execution_time', 0):.2f}s")
        click.echo(f"  Process time: {indexing.get('process_time', 0):.2f}s")
        click.echo()

    for key in ["top_10", "top_100", "top_10_filtered", "top_100_filtered"]:
        if key in results.get("results", {}):
            data = results["results"][key]
            click.echo(f"{key}:")
            click.echo(f"  Queries: {data.get('num_of_queries', 0)}")
            if "took" in data:
                click.echo(f"  Latency (ms): mean={data['took'].get('mean', 0):.2f}, "
                          f"p50={data['took'].get('50%', 0):.2f}, "
                          f"p99={data['took'].get('99%', 0):.2f}")
            if "precision" in data:
                click.echo(f"  Precision: mean={data['precision'].get('mean', 0):.4f}")
            click.echo()


if __name__ == "__main__":
    main()
