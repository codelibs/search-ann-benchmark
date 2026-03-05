---
name: upgrade-engine-version
description: Use when the user asks to upgrade, bump, or update the version of a
  vector database engine (e.g. "upgrade opensearch", "bump qdrant to latest",
  "update elasticsearch version", "エンジンのバージョンを上げて").
---

# Upgrade Engine Version

Automates version upgrades for vector database engines in the search-ann-benchmark project. Each engine version is defined in **3 locations** that must be kept in sync:

1. **Python config**: `src/search_ann_benchmark/engines/{engine}.py` — `version` field in the dataclass
2. **GitHub Actions workflows**: `.github/workflows/run-{engine}*-linux.yml` — `ENGINE_VERSION` env var
3. **README.md**: Supported Engines version table

## Engine Registry

The following engines are supported. Use `README.md`'s Supported Engines table as the source of truth for current versions, and discover file paths dynamically using Glob/Grep.

| Engine | GitHub Repository | Version Format Notes |
|--------|-------------------|----------------------|
| qdrant | qdrant/qdrant | No `v` prefix in config; Docker tag adds it. **2 workflow files** (standard + int8) |
| elasticsearch | elastic/elasticsearch | **5 workflow files** (standard, int8, int4, bbq, bbq-disk) |
| opensearch | opensearch-project/OpenSearch | **2 workflow files** (standard + faiss) |
| milvus | milvus-io/milvus | **3 version fields**: `version`, `etcd_version`, `minio_version`. Check compatibility matrix at milvus.io |
| weaviate | weaviate/weaviate | Standard semver |
| vespa | vespa-engine/vespa | CI also downloads CLI binary at same version. Calendar-like `major.build.patch` |
| pgvector | pgvector/pgvector | Format `{ver}-pg{major}` (e.g. `0.8.1-pg17`). Preserve the `-pg{N}` suffix |
| chroma | chroma-core/chroma | Standard semver |
| redisstack | redis-stack (Redis Ltd) | Format `{redis_ver}-v{build}` (e.g. `7.4.2-v2`). Check Docker Hub `redis/redis-stack-server` tags |
| vald | vdaas/vald | Includes `v` prefix (e.g. `v1.7.17`) |
| clickhouse | ClickHouse/ClickHouse | Calendar versioning `YY.M` (e.g. `25.8`). Use stable releases only |
| lancedb | lancedb/lancedb | Embedded library (no Docker). Check PyPI (`lancedb`) and GitHub releases |

## Workflow

### Step 1: Identify Engine & Current Version

1. Parse the user's request to identify the target engine name (normalize to lowercase, match against the engine registry above).
2. Read `README.md` and extract the version from the Supported Engines table for the target engine.
3. Read `src/search_ann_benchmark/engines/{engine}.py` to get the version from the Python config dataclass.
4. Use Glob to find all matching workflow files: `.github/workflows/run-{engine}*-linux.yml`
5. Read each workflow file and extract the `ENGINE_VERSION` env var value.
6. **Report the current version from all 3 locations.** If there are discrepancies, highlight them clearly and ask the user whether to fix them as part of this upgrade.

### Step 2: Discover Latest Version

1. Use WebSearch to find the latest release:
   - Search: `{engine_name} latest release site:github.com` (using the GitHub repo from the registry)
   - Also search: `{engine_name} changelog latest version`
2. Use WebFetch on the GitHub Releases page: `https://github.com/{org}/{repo}/releases`
3. **Filter out pre-releases**: Skip any version tagged as RC, alpha, beta, dev, nightly, or preview.
4. For engines with special version formats:
   - **pgvector**: The core version is separate from the `-pg{N}` suffix. Check pgvector releases for the core version, keep the existing `-pg{N}` suffix unless the user explicitly wants to change the PostgreSQL major version.
   - **redisstack**: Check Docker Hub tags for `redis/redis-stack-server` to find the latest stable tag.
   - **clickhouse**: Use only stable releases (not LTS unless currently on LTS track).
   - **lancedb**: Check both PyPI and GitHub releases; prefer the PyPI version as the canonical one.
   - **vald**: Include the `v` prefix in the version string.
   - **milvus**: Also check latest compatible etcd and minio versions from the Milvus compatibility matrix.
5. Present the discovered latest version to the user.

### Step 3: Compare Versions

1. Compare current version (from Step 1) with latest version (from Step 2).
2. Classify the change: **major**, **minor**, or **patch**.
3. If already at the latest version, inform the user and **stop**.
4. If a specific target version was requested by the user, use that instead of latest.

### Step 4: Analyze Release Notes

1. Fetch the release notes / changelog between the current and target versions:
   - Use WebFetch on `https://github.com/{org}/{repo}/releases/tag/v{version}` (or without `v` prefix as appropriate)
   - For multi-version jumps, fetch notes for each intermediate version
2. Read the engine's Python file (`src/search_ann_benchmark/engines/{engine}.py`) to understand which APIs and features are currently used.
3. Focus analysis on ANN/vector-search-relevant changes:
   - **Breaking API changes** that would affect the benchmark code
   - **New vector features** (quantization methods, index types, distance metrics)
   - **Deprecations** of features used by this project
   - **Configuration changes** (new required settings, renamed parameters)
   - **Docker image changes** (new base image, changed ports, volume paths)
4. Summarize findings for the user.

### Step 5: Propose Changes

Present a structured proposal to the user:

```
## Upgrade Proposal: {engine} {current} -> {target}

### Files to modify:
1. `src/search_ann_benchmark/engines/{engine}.py` — version: "{current}" -> "{target}"
2. `.github/workflows/run-{engine}-linux.yml` — ENGINE_VERSION: "{current}" -> "{target}"
   [list all workflow files]
3. `README.md` — Supported Engines table: {current} -> {target}

### Release Notes Summary:
- [key changes relevant to this project]

### Breaking Changes:
- [any breaking changes, or "None detected"]

### Additional Code Changes:
- [any code changes needed beyond version strings, or "None needed"]
```

**Wait for explicit user approval before proceeding.**

### Step 6: Apply Changes

After user approval:

1. **Python config** (`src/search_ann_benchmark/engines/{engine}.py`):
   - Update the `version` field in the dataclass default value.
   - For milvus, also update `etcd_version` and `minio_version` if applicable.
   - Apply any additional code changes for breaking API changes.

2. **Workflow files** (`.github/workflows/run-{engine}*-linux.yml`):
   - Update the `ENGINE_VERSION` env var in each matching workflow file.
   - For vespa, also verify the CLI download URL pattern still works with the new version.

3. **README.md**:
   - Update the version in the Supported Engines table row for this engine.
   - Keep the rest of the row (badges, links) unchanged.

### Step 7: Verify Consistency

1. Re-read all modified files to confirm the new version appears correctly.
2. Use Grep to search for any remaining occurrences of the old version string across the repository (there may be legitimate old references in result files — only flag source/config files).
3. Report the final state:

```
## Verification Results
- Python config: {new_version} ✓
- Workflow files: {list each file} ✓
- README.md: {new_version} ✓
- Stale references: {none found | list}
```

### Step 8: Commit (if requested)

If the user asks to commit:

- Stage only the modified files (do not use `git add -A`).
- Commit message format: `chore({engine}): Bump {engine} version from {old} to {new}`
- Do **not** push unless explicitly asked.
