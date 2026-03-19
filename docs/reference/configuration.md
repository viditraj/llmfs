# Configuration Reference

LLMFS works with zero configuration -- `llmfs init` is all you need. To tune behavior, create `.llmfs/config.json`:

```json
{
  "embedder": "local",
  "embedder_model": "all-MiniLM-L6-v2",
  "chunk_size_tokens": 256,
  "chunk_overlap_tokens": 50,
  "search_cache_ttl_seconds": 300,
  "auto_relate_threshold": 0.85,
  "context_manager": {
    "max_tokens": 128000,
    "evict_at": 0.70,
    "target_after_evict": 0.50
  },
  "layers": {
    "short_term": { "ttl_minutes": 60 },
    "session":    { "ttl_minutes": null },
    "knowledge":  { "ttl_minutes": null },
    "events":     { "ttl_minutes": null }
  }
}
```

## Options

| Key | Default | Description |
|-----|---------|-------------|
| `embedder` | `"local"` | `"local"` (sentence-transformers) or `"openai"` |
| `embedder_model` | `"all-MiniLM-L6-v2"` | Model name for local embedder |
| `chunk_size_tokens` | `256` | Target chunk size in tokens (prose); `512` for code |
| `chunk_overlap_tokens` | `50` | Overlap between adjacent chunks |
| `search_cache_ttl_seconds` | `300` | How long to cache search results (0 = disabled) |
| `auto_relate_threshold` | `0.85` | Auto-create `related_to` edge when similarity exceeds this |
| `context_manager.max_tokens` | `128000` | Total context window size (tokens) |
| `context_manager.evict_at` | `0.70` | Fraction of max_tokens at which eviction starts |
| `context_manager.target_after_evict` | `0.50` | Fraction of max_tokens to reach after eviction |
| `layers.short_term.ttl_minutes` | `60` | TTL for `short_term` memories |

## Using OpenAI Embeddings

```json
{
  "embedder": "openai",
  "embedder_model": "text-embedding-3-small"
}
```

```bash
export OPENAI_API_KEY=sk-...
```

!!! note
    OpenAI embeddings are higher quality for some domains but add latency and cost. The local model (22 MB, CPU-only) handles 1,000+ queries/second and is the default.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLMFS_PATH` | Override the storage directory (same as `--llmfs-path`) |
| `OPENAI_API_KEY` | Required when using `"embedder": "openai"` |
