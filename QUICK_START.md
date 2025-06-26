# Quick Start Guide

Welcome to the reorganized Recommendation System! This guide gets you up and running with the new versioned pipeline architecture.

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or using poetry (recommended)
poetry install
```

### 2. Set Environment Variables

```bash
# Copy environment template
cp configs/environments/dev.yaml.example configs/environments/dev.yaml

# Set your API keys
export OPENAI_API_KEY="your-openai-key"
export DATABASE_URL="sqlite:///recommendations.db"
```

### 3. Test the Installation

```bash
# Run the demo
python main_new.py
```

## 📋 Command Line Interface

The new CLI provides easy access to all system features:

```bash
# Show available pipelines
python scripts/utilities/cli.py pipeline list

# Get pipeline recommendations
python scripts/utilities/cli.py pipeline recommend --features reflection --priority quality

# Run a demo
python scripts/utilities/cli.py run demo --pipeline v2_rag_reflection

# Validate configurations
python scripts/utilities/cli.py config validate configs/pipelines/v1_basic_rag.yaml
```

## 🌐 API Service

Start the API server:

```bash
# Development mode
uvicorn api.v1.recommendations:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.v1.recommendations:app --workers 4 --host 0.0.0.0 --port 8000
```

Test the API:

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "AI applications in finance", "pipeline_version": "auto"}'
```

## 🔧 Configuration

The system uses hierarchical configuration:

```
configs/
├── global_config.yaml          # System-wide settings
├── environments/
│   ├── dev.yaml               # Development environment
│   └── prod.yaml              # Production environment
└── pipelines/
    ├── v1_basic_rag.yaml      # Basic RAG configuration
    └── v2_rag_reflection.yaml # Reflection RAG configuration
```

Load configuration in code:

```python
import yaml

# Load pipeline config
with open('configs/pipelines/v2_rag_reflection.yaml') as f:
    config = yaml.safe_load(f)

# Create pipeline
from pipelines.factory import PipelineFactory
pipeline = PipelineFactory.create_pipeline('v2_rag_reflection', config)
```

## 🔄 Using Pipelines

### Basic Usage

```python
from pipelines.factory import PipelineFactory

# Automatic pipeline selection
requirements = {
    'features': ['reflection'],
    'performance_priority': 'quality',
    'stability': 'stable'
}

pipeline = PipelineFactory.create_auto_pipeline(requirements, config)
result = pipeline.run(query="AI in finance", context={"sector": "banking"})

print(f"Found {len(result.recommendations)} recommendations")
```

### Manual Pipeline Selection

```python
# Create specific pipeline version
pipeline = PipelineFactory.create_pipeline('v2_rag_reflection', config)

result = pipeline.run(
    query="sustainable investment opportunities", 
    context={
        "client_sector": "finance",
        "risk_tolerance": "moderate"
    }
)

# Access results
for rec in result.recommendations:
    print(f"- {rec['title']} (confidence: {rec['confidence']:.2f})")
```

## 📊 Pipeline Versions

| Version | Features | Use Case |
|---------|----------|----------|
| **v1_basic_rag** | Basic RAG, fast | High-volume, speed critical |
| **v2_rag_reflection** | + Reflection, quality | Complex requirements, accuracy focus |
| **v3_rag_graph** | + Knowledge graphs | Advanced reasoning |
| **v4_agentic_rag** | + Multi-agent | Complex planning tasks |

### Version Selection

```python
# Compare versions
comparison = PipelineFactory.compare_versions('v1_basic_rag', 'v2_rag_reflection')
print(f"Recommendation: {comparison['recommendation']}")

# Get recommendation based on needs
recommended = PipelineFactory.get_recommended_version({
    'features': ['reflection'],
    'performance_priority': 'quality'
})
print(f"Recommended: {recommended}")
```

## 🧪 Development & Testing

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests  
pytest tests/integration/

# Pipeline validation
python scripts/utilities/cli.py pipeline validate
```

### Adding New Pipeline Version

1. **Create pipeline class:**

```python
# pipelines/v3_custom/pipeline.py
from pipelines.base.pipeline import BasePipeline

class CustomPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__(config)
        # Custom initialization
    
    def _retrieve(self, query, context):
        # Custom retrieval logic
        pass
    
    def _rank(self, query, context, candidates):
        # Custom ranking logic
        pass
```

2. **Register in factory:**

```python
# pipelines/factory.py
_PIPELINE_REGISTRY = {
    # ... existing pipelines ...
    'v3_custom': {
        'class': CustomPipeline,
        'module': 'pipelines.v3_custom.pipeline',
        # ... configuration ...
    }
}
```

3. **Create configuration:**

```yaml
# configs/pipelines/v3_custom.yaml
pipeline:
  name: "custom_pipeline"
  version: "3.0"
  description: "Custom pipeline with special features"

# Custom settings
custom_setting: value
```

## 📁 Project Structure

```
recommendation_system/
├── configs/              # 🔧 Configuration management
├── data/                # 📊 Data sources & processing
├── pipelines/           # 🔄 Versioned ML pipelines
├── llm/                 # 🤖 LLM services & prompts
├── evaluation/          # 📈 Evaluation framework
├── api/                 # 🌐 REST API endpoints
├── scripts/             # 🛠️ Operational scripts
├── monitoring/          # 📊 System monitoring
├── tests/               # 🧪 Test suites
├── docs/                # 📚 Documentation
└── notebooks/           # 📓 Analysis notebooks
```

## 🚨 Troubleshooting

### Common Issues

**Pipeline import errors:**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install in development mode
pip install -e .
```

**Configuration not found:**
```bash
# Validate config
python scripts/utilities/cli.py config validate configs/pipelines/v1_basic_rag.yaml

# Check file exists
ls -la configs/pipelines/
```

**API connection errors:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Start with debug logging
uvicorn api.v1.recommendations:app --log-level debug
```

### Getting Help

- Check the logs: `tail -f logs/recommendation_system.log`
- Validate pipelines: `python scripts/utilities/cli.py pipeline validate`
- Run health checks: `curl http://localhost:8000/health`
- View API docs: http://localhost:8000/docs

## 🎯 Next Steps

1. **Explore pipeline features:** Try different versions and compare results
2. **Configure for your environment:** Update configs for your specific needs  
3. **Set up monitoring:** Configure dashboards and alerts
4. **Add custom data sources:** Integrate your data connectors
5. **Create custom pipelines:** Build versions for your specific use cases

## 📚 Further Reading

- [API Documentation](docs/api/README.md)
- [Architecture Overview](docs/architecture/)
- [Pipeline Development Guide](docs/development/)
- [Production Deployment](docs/deployment/)

Happy recommending! 🎉 