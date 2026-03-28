# ALICE-Recommend-SaaS

Recommendation engine as a Service — collaborative filtering, content-based similarity, online training, and model management via REST API.

## Architecture

```
Client
  |
  v
API Gateway (:8222)
  |
  v
Core Engine (:8122)
  |
  +-- Similarity Engine (ANN)
  +-- Personalization Engine (MF / NCF)
  +-- Online Trainer
  +-- Model Registry
```

## Features

- Approximate nearest neighbor similarity (HNSW, FAISS-style)
- Matrix factorization and neural collaborative filtering
- Online incremental model updates from interaction streams
- Multi-model A/B testing with traffic splitting
- Cold-start handling via content features
- Explanation generation (feature importance, similar items)

## API Endpoints

### Core Engine (port 8122)

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/recommend/similar | Find items similar to a query item |
| POST | /api/v1/recommend/personalize | Generate personalized recommendations for a user |
| POST | /api/v1/recommend/train | Submit interaction data and trigger incremental training |
| GET  | /api/v1/recommend/models | List available recommendation models |
| GET  | /api/v1/recommend/stats | Return runtime statistics |
| GET  | /health | Health check |

### Example: Similar Items

```bash
curl -X POST http://localhost:8122/api/v1/recommend/similar \
  -H 'Content-Type: application/json' \
  -d '{"item_id":"item_42","top_k":10,"model":"hnsw-v1"}'
```

### Example: Personalize

```bash
curl -X POST http://localhost:8122/api/v1/recommend/personalize \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_7","top_k":20,"context":{"time_of_day":"evening"}}'
```

## License

AGPL-3.0-or-later
