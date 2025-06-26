"""
FastAPI endpoints for the Recommendation System

Provides REST API access to the versioned pipeline system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import yaml
from pathlib import Path

from pipelines.factory import PipelineFactory
from pipelines.base.pipeline import PipelineResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recommendation System API",
    description="API for LLM-based recommendation system with versioned pipelines",
    version="2.0.0"
)


class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    query: str = Field(..., description="Search query or client description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    pipeline_version: str = Field(default="auto", description="Pipeline version to use")
    max_results: int = Field(default=5, description="Maximum number of recommendations")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    pipeline_version: str
    execution_time: float
    confidence_scores: Optional[List[float]] = None


class PipelineInfo(BaseModel):
    """Pipeline information model."""
    version: str
    name: str
    description: str
    status: str
    features: Dict[str, bool]


class PipelineRequirements(BaseModel):
    """Pipeline requirements model."""
    features: List[str] = Field(default_factory=list)
    performance_priority: str = Field(default="balanced", regex="^(speed|quality|balanced)$")
    stability: str = Field(default="stable", regex="^(stable|experimental|any)$")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Recommendation System API v2.0.0",
        "version": "2.0.0",
        "features": [
            "Versioned pipelines",
            "Automatic pipeline selection", 
            "Real-time recommendations",
            "Health monitoring"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Validate pipeline factory
        versions = PipelineFactory.list_available_versions()
        factory = PipelineFactory()
        validation_results = factory.validate_all_pipelines()
        
        valid_pipelines = sum(1 for r in validation_results.values() if r['status'] == 'valid')
        total_pipelines = len(validation_results)
        
        return {
            "status": "healthy",
            "pipelines_available": len(versions),
            "pipelines_valid": f"{valid_pipelines}/{total_pipelines}",
            "api_version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/pipelines", response_model=List[PipelineInfo])
async def list_pipelines(include_experimental: bool = False):
    """List available pipeline versions."""
    try:
        versions = PipelineFactory.list_available_versions(
            include_experimental=include_experimental
        )
        
        pipeline_info = []
        for version in versions:
            info = PipelineFactory.get_pipeline_info(version)
            pipeline_info.append(PipelineInfo(**info))
        
        return pipeline_info
    
    except Exception as e:
        logger.error(f"Error listing pipelines: {e}")
        raise HTTPException(status_code=500, detail="Failed to list pipelines")


@app.get("/pipelines/{version}", response_model=PipelineInfo)
async def get_pipeline_info(version: str):
    """Get detailed information about a specific pipeline version."""
    try:
        info = PipelineFactory.get_pipeline_info(version)
        return PipelineInfo(**info)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting pipeline info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline info")


@app.post("/pipelines/recommend")
async def recommend_pipeline(requirements: PipelineRequirements):
    """Get recommended pipeline version based on requirements."""
    try:
        recommended_version = PipelineFactory.get_recommended_version(
            requirements.dict()
        )
        
        info = PipelineFactory.get_pipeline_info(recommended_version)
        return {
            "recommended_version": recommended_version,
            "pipeline_info": PipelineInfo(**info),
            "requirements": requirements.dict()
        }
    
    except Exception as e:
        logger.error(f"Error recommending pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations using the specified or auto-selected pipeline."""
    try:
        # Determine pipeline version
        if request.pipeline_version == "auto":
            requirements = {
                'performance_priority': 'balanced',
                'stability': 'stable'
            }
            pipeline_version = PipelineFactory.get_recommended_version(requirements)
            logger.info(f"Auto-selected pipeline: {pipeline_version}")
        else:
            pipeline_version = request.pipeline_version
        
        # Load configuration
        config_path = f"configs/pipelines/{pipeline_version}.yaml"
        if Path(config_path).exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = {
                'name': pipeline_version,
                'version': '1.0',
                'environment': 'api'
            }
        
        # Override max_results if specified
        if 'ranker' not in config:
            config['ranker'] = {}
        config['ranker']['max_results'] = request.max_results
        
        # Create and run pipeline
        pipeline = PipelineFactory.create_pipeline(pipeline_version, config)
        
        # Note: In a real implementation, pipeline.run() would be called here
        # For this API demo, we'll return a mock response
        mock_result = PipelineResult(
            recommendations=[
                {
                    "id": "rec_1",
                    "title": "Sample Recommendation 1", 
                    "confidence": 0.85,
                    "relevance": 0.9,
                    "pipeline_version": pipeline_version
                },
                {
                    "id": "rec_2", 
                    "title": "Sample Recommendation 2",
                    "confidence": 0.78,
                    "relevance": 0.82,
                    "pipeline_version": pipeline_version
                }
            ],
            metadata={
                "pipeline_version": pipeline_version,
                "query": request.query,
                "context_keys": list(request.context.keys()),
                "num_candidates": 20,
                "num_final": 2
            },
            version=pipeline_version,
            execution_time=1.23,
            confidence_scores=[0.85, 0.78]
        )
        
        return RecommendationResponse(
            recommendations=mock_result.recommendations,
            metadata=mock_result.metadata,
            pipeline_version=mock_result.version,
            execution_time=mock_result.execution_time,
            confidence_scores=mock_result.confidence_scores
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")


@app.post("/recommend/batch")
async def get_batch_recommendations(
    requests: List[RecommendationRequest],
    background_tasks: BackgroundTasks
):
    """Process multiple recommendation requests in batch."""
    if len(requests) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 requests per batch"
        )
    
    try:
        results = []
        for req in requests:
            # Process each request (in real implementation, this would be optimized)
            result = await get_recommendations(req)
            results.append(result)
        
        return {
            "batch_size": len(requests),
            "results": results,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")


@app.get("/pipelines/{version}/health")
async def pipeline_health_check(version: str):
    """Check health of a specific pipeline version."""
    try:
        # Load minimal config for health check
        config = {
            'name': version,
            'version': '1.0',
            'environment': 'health_check'
        }
        
        pipeline = PipelineFactory.create_pipeline(version, config)
        health = pipeline.health_check()
        
        return health
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 