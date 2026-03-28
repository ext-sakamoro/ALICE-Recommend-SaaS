use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;

#[derive(Debug, Default, Serialize)]
struct Stats {
    similar_queries: u64,
    personalize_queries: u64,
    training_runs: u64,
    model_list_queries: u64,
    total_requests: u64,
}

type AppState = Arc<Mutex<Stats>>;

// --- request / response types ---

#[derive(Debug, Deserialize)]
struct SimilarRequest {
    item_id: String,
    top_k: Option<usize>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PersonalizeRequest {
    user_id: String,
    top_k: Option<usize>,
    context: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct TrainRequest {
    interactions: Vec<serde_json::Value>,
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct ApiResponse<T: Serialize> {
    ok: bool,
    request_id: String,
    data: T,
}

fn ok<T: Serialize>(data: T) -> Json<ApiResponse<T>> {
    Json(ApiResponse {
        ok: true,
        request_id: Uuid::new_v4().to_string(),
        data,
    })
}

// --- handlers ---

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "service": "alice-recommend-saas-core" }))
}

async fn recommend_similar(
    State(state): State<AppState>,
    Json(req): Json<SimilarRequest>,
) -> (StatusCode, Json<ApiResponse<serde_json::Value>>) {
    let mut s = state.lock().unwrap();
    s.similar_queries += 1;
    s.total_requests += 1;
    let k = req.top_k.unwrap_or(10);
    let model = req.model.unwrap_or_else(|| "hnsw-v1".to_string());
    let items: Vec<serde_json::Value> = (0..k)
        .map(|i| {
            serde_json::json!({
                "item_id": format!("item_{}", i + 100),
                "score": 1.0 - (i as f64 * 0.07),
            })
        })
        .collect();
    (
        StatusCode::OK,
        ok(serde_json::json!({
            "query_item": req.item_id,
            "model": model,
            "results": items,
        })),
    )
}

async fn recommend_personalize(
    State(state): State<AppState>,
    Json(req): Json<PersonalizeRequest>,
) -> (StatusCode, Json<ApiResponse<serde_json::Value>>) {
    let mut s = state.lock().unwrap();
    s.personalize_queries += 1;
    s.total_requests += 1;
    let k = req.top_k.unwrap_or(20);
    let items: Vec<serde_json::Value> = (0..k)
        .map(|i| {
            serde_json::json!({
                "item_id": format!("item_{}", i + 200),
                "score": 1.0 - (i as f64 * 0.04),
                "reason": "collaborative_filter",
            })
        })
        .collect();
    (
        StatusCode::OK,
        ok(serde_json::json!({
            "user_id": req.user_id,
            "context": req.context,
            "results": items,
        })),
    )
}

async fn recommend_train(
    State(state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> (StatusCode, Json<ApiResponse<serde_json::Value>>) {
    let mut s = state.lock().unwrap();
    s.training_runs += 1;
    s.total_requests += 1;
    let model = req.model.unwrap_or_else(|| "ncf-v1".to_string());
    (
        StatusCode::OK,
        ok(serde_json::json!({
            "model": model,
            "interactions_ingested": req.interactions.len(),
            "job_id": Uuid::new_v4().to_string(),
            "status": "training_started",
        })),
    )
}

async fn recommend_models(
    State(state): State<AppState>,
) -> (StatusCode, Json<ApiResponse<serde_json::Value>>) {
    let mut s = state.lock().unwrap();
    s.model_list_queries += 1;
    s.total_requests += 1;
    (
        StatusCode::OK,
        ok(serde_json::json!({
            "models": [
                { "id": "hnsw-v1", "type": "ann", "status": "ready" },
                { "id": "ncf-v1", "type": "neural_cf", "status": "ready" },
                { "id": "mf-als-v2", "type": "matrix_factorization", "status": "ready" },
                { "id": "content-v1", "type": "content_based", "status": "ready" },
            ]
        })),
    )
}

async fn recommend_stats(State(state): State<AppState>) -> Json<ApiResponse<serde_json::Value>> {
    let s = state.lock().unwrap();
    ok(serde_json::json!({
        "similar_queries": s.similar_queries,
        "personalize_queries": s.personalize_queries,
        "training_runs": s.training_runs,
        "model_list_queries": s.model_list_queries,
        "total_requests": s.total_requests,
    }))
}

// --- main ---

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let state: AppState = Arc::new(Mutex::new(Stats::default()));

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/recommend/similar", post(recommend_similar))
        .route("/api/v1/recommend/personalize", post(recommend_personalize))
        .route("/api/v1/recommend/train", post(recommend_train))
        .route("/api/v1/recommend/models", get(recommend_models))
        .route("/api/v1/recommend/stats", get(recommend_stats))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8122".to_string())
        .parse()
        .unwrap_or(8122);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("alice-recommend-saas-core listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
