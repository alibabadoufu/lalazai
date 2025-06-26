# Enhanced Modular Evaluation System

A comprehensive, modular evaluation framework for AI recommendation systems with built-in LLM-as-a-Judge capabilities for qualitative assessment.

## 🚀 Features

### **Modular Architecture**
- **Separate modules** for different evaluation types
- **Factory pattern** for easy evaluator creation and management
- **Extensible design** for adding new evaluator types
- **Clean separation** of concerns for better maintainability

### **LLM-as-a-Judge Integration** 
- **State-of-the-art** LLM evaluation methodology
- **Multi-criteria assessment** with specialized judges
- **Chain-of-thought reasoning** for better accuracy
- **Bias mitigation** and consistency improvements
- **Configurable prompts** and evaluation criteria

### **Comprehensive Component Coverage**
- **LLM Feature Evaluators**: Accuracy, completeness, and relevance of extracted features
- **Recommendation Evaluators**: Traditional metrics + LLM judges for personalization and business relevance
- **Connector Evaluators**: Data ingestion quality, throughput, and reliability
- **Profile Evaluators**: Client profile completeness, freshness, and consistency

## 📁 Architecture

```
evaluation/
├── __init__.py                    # Main package exports
├── base_evaluator.py             # Abstract base class
├── offline_evaluator.py          # Offline evaluation framework
├── online_metrics.py             # Real-time metrics
├── example_usage_enhanced.py     # Comprehensive usage examples
├── README.md                     # This documentation
└── evaluators/                   # Modular evaluator implementations
    ├── __init__.py               # Evaluator package exports
    ├── factory.py                # Factory pattern for evaluator creation
    ├── llm_evaluators.py         # LLM-as-a-judge evaluators
    ├── connector_evaluators.py   # Data connector evaluation
    ├── recommendation_evaluators.py # Recommendation system evaluation
    └── profile_evaluators.py     # Client profiling evaluation
```

## 🛠 Quick Start

### Basic Usage

```python
from evaluation import EvaluatorFactory, EvaluationConfig

# Create factory
factory = EvaluatorFactory()

# Method 1: Create from configuration
config = {
    'name': 'my_llm_evaluator',
    'type': 'llm_feature',
    'enabled': True,
    'custom_parameters': {
        'enable_llm_judges': True,
        'num_judge_runs': 2
    }
}

evaluators = factory.create_evaluators_from_config([config])

# Method 2: Create individual evaluator
eval_config = EvaluationConfig(
    evaluator_name='recommendation_eval',
    enabled=True
)

rec_evaluator = factory.create_evaluator('recommendation', eval_config)
```

### LLM-as-a-Judge Evaluation

```python
from evaluation import LLMFeatureEvaluator, EvaluationConfig

# Create LLM evaluator with judges
config = EvaluationConfig(
    evaluator_name='feature_evaluator',
    enabled=True,
    custom_parameters={
        'enable_llm_judges': True,
        'num_judge_runs': 3  # Higher for more confidence
    }
)

evaluator = LLMFeatureEvaluator(config)

# Evaluation data
data = {
    'component_name': 'financial_feature_extractor',
    'input_data': {
        'content': 'Company XYZ reported Q3 revenue of $1.2B...',
        'document_type': 'earnings_report'
    },
    'extracted_features': {
        'revenue': '$1.2B',
        'quarter': 'Q3',
        'company': 'XYZ Corp'
    },
    'reference_features': {  # Ground truth
        'revenue': '$1.2B',
        'quarter': 'Q3', 
        'company': 'XYZ Corp'
    }
}

# Run evaluation with LLM judges
results = evaluator.run_evaluation(data)
```

## 📊 Evaluator Types

### 1. LLM Feature Evaluators

Evaluate LLM-based feature extraction with multiple specialized judges:

- **Accuracy Judge**: Compares extracted features against ground truth
- **Completeness Judge**: Assesses coverage of important information
- **Relevance Judge**: Evaluates relevance to client context

```python
# Features measured:
# - Feature extraction accuracy
# - Information completeness  
# - Client context relevance
# - Processing efficiency
# - Token usage optimization
```

### 2. Recommendation Evaluators

Hybrid evaluation combining traditional metrics with LLM judges:

**Traditional Metrics:**
- Relevance to client profile
- Recommendation diversity
- Interest coverage
- Ranking quality

**LLM Judges:**
- Personalization quality
- Business relevance
- Content appropriateness

```python
# Example: Recommendation evaluation
data = {
    'client_id': 'client_001',
    'client_profile': {
        'sector': 'Technology',
        'risk_tolerance': 'Moderate',
        'investment_themes': ['ESG', 'AI']
    },
    'recommendations': [
        {
            'title': 'AI Investment Opportunities',
            'sectors': ['technology'],
            'themes': ['AI'],
            'relevance_score': 0.92
        }
    ]
}
```

### 3. Connector Evaluators

Evaluate data ingestion and connector performance:

- **Success Rate**: Ingestion success percentage
- **Data Quality**: Completeness, validity, consistency, timeliness
- **Throughput**: Records processed per second
- **Error Handling**: Error rates and recovery

```python
# Configurable quality thresholds
thresholds = {
    'completeness': 0.95,
    'validity': 0.90,
    'consistency': 0.85,
    'timeliness': 0.80
}
```

### 4. Profile Evaluators  

Assess client profiling quality:

- **Completeness**: Essential vs optional field coverage
- **Freshness**: Data recency and update frequency
- **Consistency**: Cross-source data consistency
- **Confidence**: Profile attribute confidence scores

## 🔧 Advanced Configuration

### Custom LLM Judges

```python
from evaluation.evaluators.llm_evaluators import (
    LLMJudge, JudgmentCriteria, JudgmentPrompt, JudgmentType
)

# Define custom judgment criteria
criteria = JudgmentCriteria(
    name="business_impact",
    description="How likely the recommendation is to drive business value",
    judgment_type=JudgmentType.SCALE,
    scale_range=(1, 5)
)

# Create custom prompt
prompt = JudgmentPrompt(
    system_prompt="You are a financial advisor...",
    evaluation_template="Evaluate the business impact...",
    examples=[],
    output_format="JSON with score and reasoning"
)

# Create custom judge
judge = LLMJudge(llm_client, criteria, prompt, num_runs=2)
```

### Factory Pattern Configuration

```python
# Comprehensive evaluator configuration
configs = [
    {
        'name': 'production_llm_evaluator',
        'type': 'llm_feature',
        'enabled': True,
        'evaluation_interval_hours': 6,
        'sample_size': 100,
        'thresholds': {
            'accuracy': 0.85,
            'completeness': 0.90
        },
        'custom_parameters': {
            'enable_llm_judges': True,
            'num_judge_runs': 3,
            'llm_model': 'gpt-4',
            'temperature': 0.1
        }
    }
]

evaluators = factory.create_evaluators_from_config(configs)
```

## 📈 Best Practices

### LLM-as-a-Judge Implementation

1. **Multiple Runs**: Use 2-3 judge runs for consistency
2. **Clear Prompts**: Provide detailed, unambiguous instructions
3. **Structured Output**: Use JSON format for reliable parsing
4. **Chain-of-Thought**: Ask for reasoning before final judgment
5. **Bias Mitigation**: Vary prompt phrasing and examples

### Evaluation Design

1. **Ground Truth**: Establish reliable reference standards
2. **Balanced Metrics**: Combine traditional and LLM-based metrics
3. **Context Awareness**: Include relevant business and client context
4. **Threshold Tuning**: Set appropriate thresholds for your use case
5. **Regular Validation**: Periodically validate evaluator performance

### Performance Optimization

1. **Batch Processing**: Group evaluations for efficiency
2. **Caching**: Cache LLM responses for repeated evaluations
3. **Async Processing**: Use async operations for I/O-bound tasks
4. **Monitoring**: Track evaluation performance and costs

## 🔍 Monitoring and Observability

### Built-in Metrics

- Evaluation execution time
- LLM token usage and costs
- Judge consistency scores
- Error rates and types

### Custom Monitoring

```python
# Get evaluator status
info = factory.get_evaluator_info()
print(f"Active evaluators: {len(info['created_evaluators'])}")
print(f"Supported types: {info['supported_types']}")

# Per-evaluator monitoring
for name, evaluator in factory.get_all_evaluators().items():
    print(f"{name}: {evaluator.status.value}")
    print(f"Last evaluation: {evaluator.last_evaluation_time}")
```

## 🚀 Extending the System

### Adding New Evaluator Types

1. **Create Evaluator Class**: Inherit from `BaseEvaluator`
2. **Register with Factory**: Add to evaluator registry
3. **Define Configuration**: Specify required parameters
4. **Add Validation**: Implement config validation logic

```python
from evaluation.evaluators.factory import EvaluatorFactory, EvaluatorType

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, data: Dict[str, Any], **kwargs) -> List[EvaluationResult]:
        # Implementation here
        pass

# Register new evaluator type
EvaluatorFactory.register_evaluator(
    EvaluatorType.CUSTOM, 
    CustomEvaluator
)
```

### Custom LLM Integration

```python
from evaluation.llm_services.llm_client import LLMClient

class CustomLLMClient(LLMClient):
    def generate_response(self, prompt: str, **kwargs) -> str:
        # Custom LLM integration
        pass
```

## 📝 Migration Guide

### From v1.0 to v2.0

The modular architecture maintains backward compatibility:

```python
# Old way (still works)
from evaluation.component_evaluators import LLMFeatureEvaluator

# New way (recommended)
from evaluation import LLMFeatureEvaluator, EvaluatorFactory

# Factory pattern (new)
factory = EvaluatorFactory()
evaluator = factory.create_evaluator('llm_feature', config)
```

### Key Changes

1. **Modular Structure**: Evaluators moved to separate modules
2. **Factory Pattern**: New creation and management system
3. **LLM-as-a-Judge**: Enhanced with specialized judges
4. **Enhanced Configuration**: More granular control options

## 🔗 Related Documentation

- [Base Evaluator API](./base_evaluator.py)
- [LLM Services](../llm_services/README.md)
- [Example Usage](./example_usage_enhanced.py)
- [Configuration Guide](../config/evaluation_config.yaml)

## 🤝 Contributing

When adding new evaluators or judges:

1. Follow the established patterns
2. Add comprehensive tests
3. Document configuration options
4. Include usage examples
5. Update this README

---

*The enhanced modular evaluation system provides a robust, extensible foundation for comprehensive AI system assessment with state-of-the-art LLM-as-a-judge capabilities.* 