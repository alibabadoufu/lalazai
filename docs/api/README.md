---
title: "Recommendation System API"
description: "REST API for the LLM-based recommendation system with versioned pipelines"
---

# Recommendation System API

The Recommendation System API provides a REST interface for accessing the versioned pipeline system. It supports automatic pipeline selection, manual version selection, and comprehensive pipeline management.

## Quick Start

### Start the API Server

```bash
# Using uvicorn directly
uvicorn api.v1.recommendations:app --host 0.0.0.0 --port 8000

# Using the CLI
python scripts/utilities/cli.py run demo --pipeline auto
```

### Basic Request

<RequestExample>
```bash cURL
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "AI applications in finance",
    "context": {"client_sector": "banking"},
    "pipeline_version": "auto",
    "max_results": 5
  }'
```
</RequestExample>

<ResponseExample>
```json Success Response
{
  "recommendations": [
    {
      "id": "rec_1",
      "title": "Sample Recommendation 1",
      "confidence": 0.85,
      "relevance": 0.9,
      "pipeline_version": "v2_rag_reflection"
    }
  ],
  "metadata": {
    "pipeline_version": "v2_rag_reflection",
    "query": "AI applications in finance",
    "context_keys": ["client_sector"],
    "num_candidates": 20,
    "num_final": 2
  },
  "pipeline_version": "v2_rag_reflection",
  "execution_time": 1.23,
  "confidence_scores": [0.85, 0.78]
}
```
</ResponseExample>

## API Endpoints

### Health & Status

#### GET `/health`

Check the overall health of the API service.

<ResponseExample>
```json Health Response
{
  "status": "healthy",
  "pipelines_available": 4,
  "pipelines_valid": "4/4",
  "api_version": "2.0.0"
}
```
</ResponseExample>

#### GET `/`

Get basic API information.

<ResponseExample>
```json API Info
{
  "message": "Recommendation System API v2.0.0",
  "version": "2.0.0",
  "features": [
    "Versioned pipelines",
    "Automatic pipeline selection",
    "Real-time recommendations",
    "Health monitoring"
  ]
}
```
</ResponseExample>

### Pipeline Management

#### GET `/pipelines`

List all available pipeline versions.

<ParamField query="include_experimental" type="boolean" default="false">
Include experimental pipeline versions in the response.
</ParamField>

<ResponseExample>
```json Pipeline List
[
  {
    "version": "v1_basic_rag",
    "name": "Basic RAG Pipeline",
    "description": "Simple retrieval-augmented generation",
    "status": "stable",
    "features": {
      "reflection": false,
      "graph_reasoning": false,
      "multi_agent": false,
      "batch_processing": true,
      "confidence_scoring": true,
      "diversity_filtering": false
    }
  },
  {
    "version": "v2_rag_reflection",
    "name": "RAG with Reflection",
    "description": "RAG with self-reflection capabilities",
    "status": "stable",
    "features": {
      "reflection": true,
      "graph_reasoning": false,
      "multi_agent": false,
      "batch_processing": true,
      "confidence_scoring": true,
      "diversity_filtering": true
    }
  }
]
```
</ResponseExample>

#### GET `/pipelines/{version}`

Get detailed information about a specific pipeline version.

<ParamField path="version" type="string" required>
The pipeline version identifier (e.g., "v1_basic_rag", "v2_rag_reflection").
</ParamField>

<ResponseExample>
```json Pipeline Details
{
  "version": "v2_rag_reflection",
  "name": "RAG with Reflection",
  "description": "RAG with self-reflection capabilities for improved quality",
  "status": "stable",
  "features": {
    "reflection": true,
    "graph_reasoning": false,
    "multi_agent": false,
    "batch_processing": true,
    "confidence_scoring": true,
    "diversity_filtering": true
  }
}
```
</ResponseExample>

#### POST `/pipelines/recommend`

Get a recommended pipeline version based on requirements.

<ParamField body="features" type="array" default="[]">
List of required features (e.g., ["reflection", "graph_reasoning"]).
</ParamField>

<ParamField body="performance_priority" type="string" default="balanced">
Performance priority: "speed", "quality", or "balanced".
</ParamField>

<ParamField body="stability" type="string" default="stable">
Stability requirement: "stable", "experimental", or "any".
</ParamField>

<RequestExample>
```json Pipeline Recommendation Request
{
  "features": ["reflection"],
  "performance_priority": "quality",
  "stability": "stable"
}
```
</RequestExample>

<ResponseExample>
```json Recommendation Response
{
  "recommended_version": "v2_rag_reflection",
  "pipeline_info": {
    "version": "v2_rag_reflection",
    "name": "RAG with Reflection",
    "description": "RAG with self-reflection capabilities",
    "status": "stable",
    "features": {
      "reflection": true,
      "confidence_scoring": true,
      "diversity_filtering": true
    }
  },
  "requirements": {
    "features": ["reflection"],
    "performance_priority": "quality",
    "stability": "stable"
  }
}
```
</ResponseExample>

#### GET `/pipelines/{version}/health`

Check the health of a specific pipeline version.

<ParamField path="version" type="string" required>
The pipeline version to check.
</ParamField>

<ResponseExample>
```json Pipeline Health
{
  "status": "healthy",
  "version": "v2_rag_reflection",
  "components": {
    "retriever": "healthy",
    "ranker": "healthy",
    "reflection_agent": "healthy"
  },
  "last_check": "2024-01-15T10:30:00Z"
}
```
</ResponseExample>

### Recommendations

#### POST `/recommend`

Generate recommendations using the specified or auto-selected pipeline.

<ParamField body="query" type="string" required>
The search query or client description to find recommendations for.
</ParamField>

<ParamField body="context" type="object" default="{}">
Additional context information to improve recommendations.
</ParamField>

<ParamField body="pipeline_version" type="string" default="auto">
Pipeline version to use. Use "auto" for automatic selection based on optimal performance.
</ParamField>

<ParamField body="max_results" type="integer" default="5">
Maximum number of recommendations to return (1-50).
</ParamField>

<RequestExample>
```json Basic Recommendation Request
{
  "query": "sustainable investment opportunities",
  "context": {
    "client_sector": "finance",
    "risk_tolerance": "moderate",
    "investment_horizon": "long_term"
  },
  "pipeline_version": "v2_rag_reflection",
  "max_results": 3
}
```
</RequestExample>

<ResponseExample>
```json Recommendation Response
{
  "recommendations": [
    {
      "id": "rec_1",
      "title": "ESG Bond Portfolio Strategy",
      "confidence": 0.92,
      "relevance": 0.88,
      "pipeline_version": "v2_rag_reflection",
      "description": "Diversified ESG-focused bond strategy...",
      "tags": ["ESG", "Bonds", "Sustainable"]
    },
    {
      "id": "rec_2", 
      "title": "Green Technology ETF Analysis",
      "confidence": 0.87,
      "relevance": 0.85,
      "pipeline_version": "v2_rag_reflection",
      "description": "Comprehensive analysis of green tech ETFs...",
      "tags": ["ETF", "Technology", "Green"]
    }
  ],
  "metadata": {
    "pipeline_version": "v2_rag_reflection",
    "query": "sustainable investment opportunities",
    "context_keys": ["client_sector", "risk_tolerance", "investment_horizon"],
    "num_candidates": 45,
    "num_final": 3,
    "reflection_iterations": 2,
    "confidence_improvement": 0.15
  },
  "pipeline_version": "v2_rag_reflection", 
  "execution_time": 2.34,
  "confidence_scores": [0.92, 0.87, 0.81]
}
```
</ResponseExample>

#### POST `/recommend/batch`

Process multiple recommendation requests in a single batch operation.

<ParamField body="requests" type="array" required>
Array of recommendation requests (maximum 100 per batch).
</ParamField>

<RequestExample>
```json Batch Request
[
  {
    "query": "AI in healthcare",
    "context": {"sector": "healthcare"},
    "pipeline_version": "auto",
    "max_results": 3
  },
  {
    "query": "blockchain applications",
    "context": {"sector": "fintech"},
    "pipeline_version": "v2_rag_reflection",
    "max_results": 5
  }
]
```
</RequestExample>

<ResponseExample>
```json Batch Response
{
  "batch_size": 2,
  "results": [
    {
      "recommendations": [...],
      "metadata": {...},
      "pipeline_version": "v2_rag_reflection",
      "execution_time": 1.45
    },
    {
      "recommendations": [...], 
      "metadata": {...},
      "pipeline_version": "v2_rag_reflection",
      "execution_time": 2.12
    }
  ],
  "status": "completed"
}
```
</ResponseExample>

## Pipeline Versions

### v1_basic_rag

<Card title="Basic RAG Pipeline" icon="zap">
Simple, fast retrieval-augmented generation for baseline performance.
</Card>

**Features:**
- ✅ Batch processing
- ✅ Confidence scoring  
- ❌ Reflection
- ❌ Graph reasoning
- ❌ Multi-agent

**Use Cases:**
- High-volume requests where speed is critical
- Baseline comparisons
- Simple recommendation scenarios

### v2_rag_reflection

<Card title="RAG with Reflection" icon="refresh">
Enhanced pipeline with self-reflection for improved quality.
</Card>

**Features:**
- ✅ Batch processing
- ✅ Confidence scoring
- ✅ Reflection capabilities
- ✅ Diversity filtering
- ❌ Graph reasoning
- ❌ Multi-agent

**Use Cases:**
- Quality-focused recommendations
- Complex client requirements
- When accuracy is more important than speed

### v3_rag_graph (Coming Soon)

<Card title="RAG with Knowledge Graph" icon="share-nodes">
Advanced reasoning using knowledge graph integration.
</Card>

**Features:**
- ✅ All v2 features
- ✅ Graph reasoning
- ✅ Entity relationship analysis
- ❌ Multi-agent

### v4_agentic_rag (Coming Soon)

<Card title="Multi-Agent RAG" icon="users">
Sophisticated multi-agent system for complex reasoning tasks.
</Card>

**Features:**
- ✅ All v3 features  
- ✅ Multi-agent coordination
- ✅ Advanced planning
- ✅ Dynamic workflow adaptation

## Error Handling

### Error Response Format

<ResponseExample>
```json Error Response
{
  "detail": "Pipeline version 'invalid_version' not found",
  "status_code": 404,
  "timestamp": "2024-01-15T10:30:00Z"
}
```
</ResponseExample>

### Common Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| **400** | Bad Request | Invalid pipeline version, malformed request |
| **404** | Not Found | Pipeline version doesn't exist |
| **422** | Validation Error | Invalid request parameters |
| **500** | Internal Error | Pipeline execution failure, system error |
| **503** | Service Unavailable | System unhealthy, dependencies down |

## Authentication

<Info>
The current version runs without authentication for development purposes. 
Production deployments should implement proper authentication (API keys, JWT tokens, etc.).
</Info>

For production use, add authentication headers:

<RequestExample>
```bash cURL with Auth
curl -X POST "http://localhost:8000/recommend" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "AI applications", "pipeline_version": "auto"}'
```
</RequestExample>

## Rate Limiting

<Warning>
The API currently has no rate limiting. Production deployments should implement appropriate rate limiting based on usage patterns.
</Warning>

Recommended limits:
- **Standard requests**: 100 requests/minute
- **Batch requests**: 10 requests/minute  
- **Pipeline management**: 20 requests/minute

## Monitoring

### Health Checks

Monitor these endpoints for system health:

- `GET /health` - Overall system health
- `GET /pipelines/{version}/health` - Specific pipeline health

### Metrics to Track

- Request latency (p50, p95, p99)
- Request volume by pipeline version
- Error rates by endpoint
- Pipeline execution times
- Confidence score distributions

### Logging

The API logs important events:

```bash
# Start the API with verbose logging
uvicorn api.v1.recommendations:app --log-level debug

# View logs
tail -f /var/log/recommendation-system/api.log
```

## Development

### Running Locally

<Steps>
<Step title="Install dependencies">
```bash
pip install -r requirements.txt
```
</Step>

<Step title="Start the API server">
```bash
uvicorn api.v1.recommendations:app --reload --host 0.0.0.0 --port 8000
```
</Step>

<Step title="Test the API">
```bash
curl http://localhost:8000/health
```

<Check>
You should see a healthy status response.
</Check>
</Step>
</Steps>

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Testing

<Tabs>
<Tab title="Unit Tests">
```bash
pytest tests/unit/api/
```
</Tab>

<Tab title="Integration Tests">
```bash
pytest tests/integration/api/
```
</Tab>

<Tab title="Load Testing">
```bash
# Using Apache Bench
ab -n 1000 -c 10 -T 'application/json' \
   -p test_request.json \
   http://localhost:8000/recommend
```
</Tab>
</Tabs> 