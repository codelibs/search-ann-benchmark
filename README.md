# Search ANN Benchmark

Benchmark the search performance of Approximate Nearest Neighbor (ANN) algorithms implemented in various systems.
This repository contains a Python CLI tool to evaluate and compare the efficiency and accuracy of ANN searches across different platforms.

## Introduction

Approximate Nearest Neighbor (ANN) search algorithms are essential for handling high-dimensional data spaces, enabling fast and resource-efficient retrieval of similar items from large datasets.
This benchmarking suite aims to provide an empirical basis for comparing the performance of several popular ANN-enabled search systems.

## Supported Engines

| Engine | Version | GitHub Actions |
|--------|---------|----------------|
| Qdrant | 1.16.2 | [![Run Qdrant](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-qdrant-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-qdrant-linux.yml) |
| Elasticsearch | 9.2.3 | [![Run Elasticsearch](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-elasticsearch-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-elasticsearch-linux.yml) |
| OpenSearch | 3.4.0 | [![Run OpenSearch](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-opensearch-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-opensearch-linux.yml) |
| Milvus | 2.6.7 | [![Run Milvus](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-milvus-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-milvus-linux.yml) |
| Weaviate | 1.35.2 | [![Run Weaviate](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-weaviate-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-weaviate-linux.yml) |
| Vespa | 8.620.35 | [![Run Vespa](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-vespa-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-vespa-linux.yml) |
| pgvector | 0.8.1-pg17 | [![Run pgvector](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-pgvector-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-pgvector-linux.yml) |
| Chroma | 1.4.0 | [![Run Chroma](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-chroma-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-chroma-linux.yml) |
| Redis Stack | 7.4.2-v2 | [![Run Redis Stack](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-redisstack-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-redisstack-linux.yml) |
| Vald | 1.7.13 | [![Run Vald](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-vald-linux.yml/badge.svg)](https://github.com/codelibs/search-ann-benchmark/actions/workflows/run-vald-linux.yml) |

## Prerequisites

Before running the benchmarks, ensure you have the following installed:

- Docker
- Python 3.10 or higher
- uv (Python package manager)

## Installation

1. **Install uv (if not already installed):**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **Clone the repository and install dependencies:**

    ```bash
    git clone https://github.com/codelibs/search-ann-benchmark.git
    cd search-ann-benchmark
    uv sync
    ```

3. **Download the dataset:**

    ```bash
    bash scripts/setup.sh
    ```

    For GitHub Actions (smaller dataset):
    ```bash
    bash scripts/setup.sh gha
    ```

## Usage

### Run a benchmark

```bash
# Run Qdrant benchmark with default settings
uv run search-ann-benchmark run qdrant

# Run Elasticsearch with specific configuration
uv run search-ann-benchmark run elasticsearch --target 1m-768-m48-efc200-ef100-ip

# Run with quantization
uv run search-ann-benchmark run elasticsearch --quantization int8 --variant int8

# Skip filtered search benchmark
uv run search-ann-benchmark run chroma --no-filter
```

### List available engines

```bash
uv run search-ann-benchmark list-engines
```

### List available configurations

```bash
uv run search-ann-benchmark list-targets
```

### Show configuration details

```bash
uv run search-ann-benchmark show-config qdrant --target 100k-768-m32-efc200-ef100-ip
```

### View benchmark results

```bash
uv run search-ann-benchmark show-results results.json
```

## Configuration Options

### Target Configurations

| Name | Index Size | HNSW M | Description |
|------|------------|--------|-------------|
| 100k-768-m32-efc200-ef100-ip | 100,000 | 32 | Small dataset for quick testing |
| 1m-768-m48-efc200-ef100-ip | 1,000,000 | 48 | Medium dataset |
| 5m-768-m48-efc200-ef100-ip | 5,000,000 | 48 | Full dataset |

### Quantization Options

Different engines support different quantization modes:

- **Qdrant**: none, int8
- **Elasticsearch**: none, int4, int8, bbq
- **OpenSearch**: none (supports faiss engine variant)
- **Weaviate**: none, pq
- **pgvector**: vector, halfvec

## Project Structure

```
search-ann-benchmark/
├── src/search_ann_benchmark/
│   ├── __init__.py
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration classes
│   ├── runner.py           # Benchmark orchestration
│   ├── core/
│   │   ├── base.py         # Abstract engine interface
│   │   ├── docker.py       # Docker management
│   │   ├── embedding.py    # Embedding loader
│   │   └── metrics.py      # Metrics calculation
│   └── engines/
│       ├── qdrant.py
│       ├── elasticsearch.py
│       ├── opensearch.py
│       ├── milvus.py
│       ├── weaviate.py
│       ├── vespa.py
│       ├── pgvector.py
│       ├── chroma.py
│       ├── redisstack.py
│       └── vald.py
├── tests/
├── scripts/
│   ├── setup.sh            # Dataset download
│   └── get_hardware_info.sh
└── .github/workflows/      # CI workflows
```

## Output Format

Benchmark results are saved to `results.json` with the following structure:

```json
{
  "variant": "",
  "target": "100k-768-m32-efc200-ef100-ip",
  "version": "1.13.6",
  "settings": { ... },
  "results": {
    "indexing": {
      "execution_time": 123.45,
      "process_time": 100.23,
      "container": { ... }
    },
    "top_10": {
      "num_of_queries": 10000,
      "took": { "mean": 5.2, "std": 1.1, ... },
      "hits": { ... },
      "precision": { "mean": 0.95, ... }
    },
    "top_100": { ... },
    "top_10_filtered": { ... },
    "top_100_filtered": { ... }
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

## Development

### Running tests

```bash
uv run pytest
```

### Code formatting

```bash
uv run ruff check --fix src tests
uv run ruff format src tests
```

### Type checking

```bash
uv run mypy src
```

## Updating Engine Versions

To update an engine version, modify the `ENGINE_VERSION` in:
1. The engine config class in `src/search_ann_benchmark/engines/<engine>.py`
2. The corresponding workflow in `.github/workflows/run-<engine>-linux.yml`

## Benchmark Results

For a comparison of the results, including response times and precision metrics for different ANN algorithms, see [Benchmark Results Page](https://codelibs.co/benchmark/ann-benchmark.html).

## Contributing

We welcome contributions!
If you have suggestions for additional benchmarks, improvements to existing ones, or fixes for any issues, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the Apache License 2.0.
