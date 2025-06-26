"""
Main entry point for the Intelligent Research Recommendation System.
This module orchestrates the complete recommendation pipeline as specified in the PRD.
"""

import logging
import sys
import json
from typing import Dict, Any, List
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recommendation_system.log')
    ]
)

from core_pipeline.recommendation.orchestrator import Orchestrator
from data_connectors.bloomberg_chat.connector import BloombergChatConnector
from data_connectors.refinitiv_chat.connector import RefinitivChatConnector
from data_connectors.publications_connector import PublicationsConnector
from data_connectors.crm_connector import CRMConnector
from data_connectors.rfq_connector import RFQConnector
from data_connectors.kyc_connector import KYCConnector
from data_connectors.readership_connector import ReadershipConnector
from core_pipeline.email_generator import EmailGenerator
from llm_services.llm_client import LLMClient
from llm_services.prompt_manager import PromptManager
from data_connectors.schemas import PublicationData, PublicationType


def load_sample_publications() -> Dict[str, Dict[str, Any]]:
    """
    Load sample publications data.
    In production, this would connect to the publications database/API.
    
    Returns:
        Dictionary of sample publications
    """
    return {
        "pub1": {
            "id": "pub1",
            "title": "The Future of AI in Finance: Transforming Trading and Risk Management",
            "authors": ["Dr. John Doe", "Sarah Chen"],
            "date": "2025-06-20",
            "full_text": """
            This comprehensive analysis explores how artificial intelligence is revolutionizing 
            the financial services industry. We examine the latest developments in algorithmic 
            trading, risk assessment, and client relationship management. Key findings include 
            a 40% improvement in trade execution efficiency and significant reduction in 
            operational risks. The report covers machine learning applications in credit 
            scoring, fraud detection, and regulatory compliance. We recommend that financial 
            institutions invest heavily in AI capabilities to remain competitive.
            """,
            "metadata": {"type": "deep_dive", "sector": "technology", "priority": "high"}
        },
        "pub2": {
            "id": "pub2", 
            "title": "Weekly Market Outlook: Global Economic Indicators",
            "authors": ["Jane Smith"],
            "date": "2025-06-23",
            "full_text": """
            This week's market analysis focuses on key economic indicators and their 
            impact on global markets. US employment data shows continued strength with 
            unemployment at historic lows. European markets are showing resilience 
            despite ongoing geopolitical concerns. Asian markets are experiencing 
            volatility due to supply chain disruptions. We maintain a cautiously 
            optimistic outlook for the remainder of Q2 2025.
            """,
            "metadata": {"type": "weekly_report", "priority": "medium"}
        },
        "pub3": {
            "id": "pub3",
            "title": "Shipping Costs Analysis: Supply Chain Disruption Impact",
            "authors": ["Peter Jones", "Maria Rodriguez"],
            "date": "2025-06-19", 
            "full_text": """
            Global shipping costs have surged 65% year-over-year, creating significant 
            challenges for international trade. This report analyzes the root causes 
            including port congestion, container shortages, and fuel price increases. 
            We examine the impact on various industries and provide recommendations 
            for supply chain optimization. Technology solutions including AI-powered 
            logistics platforms show promise for reducing costs by 20-30%. Companies 
            should consider diversifying shipping routes and investing in automation.
            """,
            "metadata": {"type": "deep_dive", "sector": "logistics", "priority": "high"}
        },
        "pub4": {
            "id": "pub4",
            "title": "Morning Comment: Tech Sector Volatility",
            "authors": ["Alice Brown"],
            "date": "2025-06-24",
            "full_text": """
            Tech stocks opened lower this morning following concerns about regulatory 
            changes and interest rate movements. Key movers include major cloud 
            computing providers and semiconductor companies. We expect continued 
            volatility as markets digest recent earnings reports. Traders should 
            monitor support levels and consider defensive positioning.
            """,
            "metadata": {"type": "morning_comment", "sector": "technology", "priority": "urgent"}
        }
    }


def load_sample_salesperson_config() -> Dict[str, Any]:
    """
    Load sample salesperson and client configuration.
    In production, this would come from CRM/user management systems.
    
    Returns:
        Dictionary with salesperson and client information
    """
    return {
        "salesperson_name": "Alex Thompson",
        "clients": [
            {
                "client_id": "Chen",
                "company_name": "Global Logistics Corp",
                "sector": "logistics",
                "tier": "tier1"
            },
            {
                "client_id": "Maria", 
                "company_name": "TechVenture Capital",
                "sector": "technology",
                "tier": "tier2"
            }
        ]
    }


def initialize_system() -> tuple:
    """
    Initialize all system components.
    
    Returns:
        Tuple of (orchestrator, email_generator, llm_client, prompt_manager)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = LLMClient(
            model="gpt-4",
            max_retries=3,
            timeout=60,
            token_budget=50000  # Conservative daily budget
        )
        
        # Test LLM client health
        health_status = llm_client.health_check()
        logger.info(f"LLM Client Status: {health_status['status']}")
        
        # Initialize prompt manager
        logger.info("Initializing prompt manager...")
        prompt_manager = PromptManager()
        
        # Initialize orchestrator
        logger.info("Initializing recommendation orchestrator...")
    orchestrator = Orchestrator()
        orchestrator.set_llm_client(llm_client)
        
        # Initialize email generator
        logger.info("Initializing email generator...")
        email_generator = EmailGenerator(llm_client, prompt_manager)
        
        logger.info("System initialization completed successfully")
        return orchestrator, email_generator, llm_client, prompt_manager
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise


def register_data_connectors(orchestrator: Orchestrator, llm_client: LLMClient) -> None:
    """
    Register all data connectors with the orchestrator.
    
    Args:
        orchestrator: Main orchestrator instance
        llm_client: LLM client for processing
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Register Bloomberg Chat connector
        bloomberg_config = {
            "connector_name": "bloomberg_chat",
            "enabled": True,
            "lookback_days": 90,
            "api_base_url": os.getenv("BLOOMBERG_API_URL"),
            "api_key": os.getenv("BLOOMBERG_API_KEY"),
            "timeout": 30
        }
        bloomberg_connector = BloombergChatConnector(bloomberg_config)
        orchestrator.profiling.register_connector("bloomberg_chat", bloomberg_connector)
        logger.info("Registered Bloomberg Chat connector")
        
        # Register Refinitiv Chat connector
        refinitiv_config = {
            "connector_name": "refinitiv_chat",
            "enabled": True,
            "lookback_days": 90,
            "api_base_url": os.getenv("REFINITIV_API_URL"),
            "api_key": os.getenv("REFINITIV_API_KEY"),
            "app_key": os.getenv("REFINITIV_APP_KEY"),
            "timeout": 30
        }
        refinitiv_connector = RefinitivChatConnector(refinitiv_config)
        orchestrator.profiling.register_connector("refinitiv_chat", refinitiv_connector)
        logger.info("Registered Refinitiv Chat connector")
        
        # Register CRM connector
        crm_config = {
            "connector_name": "crm",
            "enabled": True,
            "lookback_days": 180,
            "crm_type": "salesforce",
            "api_base_url": os.getenv("CRM_API_URL"),
            "api_key": os.getenv("CRM_API_KEY"),
            "database_path": os.getenv("CRM_DATABASE_PATH"),
            "timeout": 30
        }
        crm_connector = CRMConnector(crm_config)
        orchestrator.profiling.register_connector("crm", crm_connector)
        logger.info("Registered CRM connector")
        
        # Register RFQ connector
        rfq_config = {
            "connector_name": "rfq",
            "enabled": True,
            "lookback_days": 30,
            "trading_system_url": os.getenv("TRADING_SYSTEM_URL"),
            "api_key": os.getenv("TRADING_API_KEY"),
            "database_path": os.getenv("RFQ_DATABASE_PATH"),
            "timeout": 30
        }
        rfq_connector = RFQConnector(rfq_config)
        orchestrator.profiling.register_connector("rfq", rfq_connector)
        logger.info("Registered RFQ connector")
        
        # Register KYC connector
        kyc_config = {
            "connector_name": "kyc",
            "enabled": True,
            "kyc_system_url": os.getenv("KYC_SYSTEM_URL"),
            "api_key": os.getenv("KYC_API_KEY"),
            "database_path": os.getenv("KYC_DATABASE_PATH"),
            "timeout": 30
        }
        kyc_connector = KYCConnector(kyc_config)
        orchestrator.profiling.register_connector("kyc", kyc_connector)
        logger.info("Registered KYC connector")
        
        # Register Readership connector
        readership_config = {
            "connector_name": "readership",
            "enabled": True,
            "lookback_days": 60,
            "analytics_url": os.getenv("ANALYTICS_URL"),
            "api_key": os.getenv("ANALYTICS_API_KEY"),
            "database_path": os.getenv("READERSHIP_DATABASE_PATH"),
            "log_files_path": os.getenv("LOG_FILES_PATH"),
            "timeout": 30
        }
        readership_connector = ReadershipConnector(readership_config)
        orchestrator.profiling.register_connector("readership", readership_connector)
        logger.info("Registered Readership connector")
        
        # Register Publications connector
        publications_config = {
            "connector_name": "publications", 
            "enabled": True,
            "lookback_days": 30
        }
        publications_connector = PublicationsConnector(publications_config)
        orchestrator.profiling.register_connector("publications", publications_connector)
        logger.info("Registered Publications connector")
        
        # Set LLM client for profiling
        orchestrator.profiling.set_llm_client(llm_client)
        
        logger.info("All data connectors registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering data connectors: {e}")
        raise


def run_recommendation_pipeline(orchestrator: Orchestrator, 
                              publications: Dict[str, Dict[str, Any]], 
                              client_config: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Run the recommendation pipeline for all clients.
    
    Args:
        orchestrator: Main orchestrator instance
        publications: Dictionary of publications
        client_config: Client configuration
        
    Returns:
        Dictionary of recommendations by client
    """
    logger = logging.getLogger(__name__)
    
    try:
        client_recommendations = {}
        
        for client_info in client_config["clients"]:
            client_id = client_info["client_id"]
            
            logger.info(f"Running recommendation pipeline for client: {client_id}")
            
            # Run the pipeline
            recommendations = orchestrator.run_pipeline(client_id, publications)
            client_recommendations[client_id] = recommendations
            
            logger.info(f"Generated {len(recommendations)} recommendations for {client_id}")
        
        return client_recommendations
        
    except Exception as e:
        logger.error(f"Error running recommendation pipeline: {e}")
        raise


def generate_email_reports(email_generator: EmailGenerator,
                          salesperson_name: str,
                          client_recommendations: Dict[str, List[Any]]) -> str:
    """
    Generate email reports for the salesperson.
    
    Args:
        email_generator: Email generator instance
        salesperson_name: Name of the salesperson
        client_recommendations: Recommendations by client
        
    Returns:
        Generated email content
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Generating email report for {salesperson_name}")
        
        # Generate the email report
        email_content = email_generator.generate_email_report(
            salesperson_name, 
            client_recommendations
        )
        
        logger.info("Email report generated successfully")
        return email_content
        
    except Exception as e:
        logger.error(f"Error generating email report: {e}")
        raise


def get_system_health_status(orchestrator: Orchestrator, 
                           llm_client: LLMClient) -> Dict[str, Any]:
    """
    Get comprehensive system health status.
    
    Args:
        orchestrator: Main orchestrator instance
        llm_client: LLM client instance
        
    Returns:
        System health status dictionary
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "system_status": "operational",
        "llm_client": llm_client.health_check(),
        "profiling_system": orchestrator.profiling.get_profiling_stats(),
        "connectors": orchestrator.profiling.get_connector_health_status()
    }


def main():
    """Main function that orchestrates the entire recommendation system."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=" * 60)
        logger.info("Starting Intelligent Research Recommendation System")
        logger.info("=" * 60)
        
        # Initialize system components
        orchestrator, email_generator, llm_client, prompt_manager = initialize_system()
        
        # Register data connectors
        register_data_connectors(orchestrator, llm_client)
        
        # Load sample data (in production, this would come from databases/APIs)
        logger.info("Loading sample publications and client data...")
        publications = load_sample_publications()
        client_config = load_sample_salesperson_config()
        
        logger.info(f"Loaded {len(publications)} publications")
        logger.info(f"Processing for salesperson: {client_config['salesperson_name']}")
        logger.info(f"Number of clients: {len(client_config['clients'])}")
        
        # Run recommendation pipeline
        logger.info("Running recommendation pipeline...")
        client_recommendations = run_recommendation_pipeline(
            orchestrator, 
            publications, 
            client_config
        )
        
        # Generate email reports
        logger.info("Generating email reports...")
        email_content = generate_email_reports(
            email_generator,
            client_config["salesperson_name"],
            client_recommendations
        )
        
        # Get system health status
        health_status = get_system_health_status(orchestrator, llm_client)
        
        # Output results
        logger.info("=" * 60)
        logger.info("RECOMMENDATION SYSTEM RESULTS")
        logger.info("=" * 60)
        
        # Print summary statistics
        total_recommendations = sum(len(recs) for recs in client_recommendations.values())
        logger.info(f"Total recommendations generated: {total_recommendations}")
        
        for client_id, recs in client_recommendations.items():
            logger.info(f"  {client_id}: {len(recs)} recommendations")
        
        # Print system health summary
        logger.info(f"System Status: {health_status['system_status']}")
        logger.info(f"LLM Client Status: {health_status['llm_client']['status']}")
        
        # Save results to file
        results = {
            "timestamp": datetime.now().isoformat(),
            "salesperson": client_config["salesperson_name"],
            "client_recommendations": client_recommendations,
            "email_content": email_content,
            "system_health": health_status
        }
        
        output_file = f"recommendation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info("Recommendation system completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in recommendation system: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


