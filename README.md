# Intelligent Research Recommendation Engine

This project implements an intelligent, automated recommendation system designed to proactively surface the most relevant research publications for sales teams and their clients. It leverages a Large Language Model (LLM) to deeply understand client needs and research content, aiming to enhance client interactions, increase research engagement, and drive transaction flow.

## Project Structure

```
/recommendation_system/
├── config/
│   ├── pipeline_config.yaml  # Main settings, thresholds, scoring weights
│   ├── prompts.yaml          # Mapping of tasks to prompt file versions
│   └── business_rules.yaml   # Lookback periods, publication expiry rules
├── data_connectors/
│   ├── base_connector.py     # Abstract Base Class for all connectors
│   ├── bloomberg_chat_connector.py # Dummy connector for Bloomberg Chat data
│   └── publications_connector.py   # Dummy connector for Publications data
├── core_pipeline/
│   ├── profiling.py          # Aggregates features from connectors into profiles
│   ├── recommendation/
│   │   ├── retriever.py      # Candidate generation from vector store
│   │   ├── ranker.py         # LLM-based scoring and evidence generation
│   │   └── orchestrator.py   # Runs the full recs pipeline
│   └── email_generator.py    # Composes and dispatches the final email
├── llm_services/
│   ├── llm_client.py         # Unified client for on-prem OpenAI compatible API
│   └── prompt_manager.py     # Manages loading and versioning of all prompts
├── prompts/
│   ├── email_composition.txt # Prompt template for email generation
│   └── relevance_scoring.txt # Prompt template for relevance scoring
├── evaluation/
│   ├── offline_evaluator.py  # Framework for testing components
│   └── online_metrics.py     # Tools for tracking live metrics
├── orchestration/
│   └── dags/                 # Airflow/Prefect DAG definitions (placeholder)
└── main.py                   # Main entrypoint
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd recommendation_system
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure LLM API (if using a real LLM):**
    Set `OPENAI_API_KEY` and `OPENAI_API_BASE` environment variables if you are using an OpenAI-compatible API. For this dummy implementation, it's not strictly necessary, but good practice for future integration.

    ```bash
    export OPENAI_API_KEY="your_api_key"
    export OPENAI_API_BASE="http://your_llm_api_endpoint/v1"
    ```

## Running the Project

To run the simulated recommendation pipeline and generate a dummy email report, execute the `main.py` script:

```bash
python3 main.py
```

This will print the simulated LLM interactions and the generated email report to the console.

## Configuration

-   `config/pipeline_config.yaml`: Adjust recommendation thresholds and scoring weights.
-   `config/business_rules.yaml`: Modify lookback periods for data connectors and publication expiry rules.
-   `config/prompts.yaml`: Define which prompt files and versions to use for different LLM tasks.
-   `prompts/`: Edit the `.txt` files to refine the LLM prompts.

## Extending the Project

-   **Add New Data Connectors:** Create new Python files in `data_connectors/` that inherit from `BaseConnector` to integrate additional data sources (e.g., CRM, RFQ, Client KYC, Readership Data).
-   **Integrate Real LLM:** Modify `llm_services/llm_client.py` to connect to your actual LLM provider (e.g., OpenAI, HuggingFace, custom on-prem model).
-   **Implement Vector Store:** Replace the dummy `publication_vector_store` in `core_pipeline/recommendation/retriever.py` with a proper vector database (e.g., Pinecone, Weaviate, FAISS).
-   **Develop Evaluation Framework:** Enhance `evaluation/offline_evaluator.py` and `evaluation/online_metrics.py` for robust model evaluation.
-   **Orchestration:** Integrate with workflow orchestration tools like Airflow or Prefect by defining DAGs in `orchestration/dags/`.


