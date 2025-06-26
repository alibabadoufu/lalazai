"""
CRM (Customer Relationship Management) Connector Implementation.
This module handles CRM data ingestion for client profiles, interactions, and relationship history.
"""

import os
import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from .base_connector import BaseConnector
from .schemas import CRMEntry, ConnectorConfig


class CRMConnector(BaseConnector):
    """
    Connector for CRM data including client profiles, interactions, and relationship history.
    
    Supports multiple CRM systems including Salesforce, HubSpot, and custom databases.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CRM connector.
        
        Args:
            config: Configuration dictionary containing CRM settings
        """
        # Set default connector name if not provided
        if "connector_name" not in config:
            config["connector_name"] = "crm"
            
        super().__init__(config)
        
        # CRM configuration
        self.crm_type = config.get("crm_type", "salesforce")  # salesforce, hubspot, database
        self.api_base_url = config.get("api_base_url", os.getenv("CRM_API_URL"))
        self.api_key = config.get("api_key", os.getenv("CRM_API_KEY"))
        self.database_path = config.get("database_path", os.getenv("CRM_DATABASE_PATH"))
        self.timeout = config.get("timeout", 30)
        
        # Database connection for local CRM data
        self.db_connection = None
        
        if not self.api_base_url and not self.database_path:
            self.logger.warning("No CRM data source configured. Using mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            
        if self.database_path:
            self._init_database()

    def _init_database(self):
        """Initialize database connection for local CRM data."""
        try:
            self.db_connection = sqlite3.connect(self.database_path)
            self.db_connection.row_factory = sqlite3.Row  # Enable column access by name
            self.logger.info(f"Connected to CRM database: {self.database_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to CRM database: {str(e)}")
            self.use_mock_data = True

    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest CRM data.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional parameters (lookback_days, interaction_type, etc.)
            
        Returns:
            List of raw CRM records
            
        Raises:
            ConnectionError: If unable to connect to CRM system
            ValueError: If invalid parameters provided
        """
        try:
            if self.use_mock_data:
                return self._get_mock_data(client_id)
            
            if self.database_path:
                return self._fetch_from_database(client_id, **kwargs)
            else:
                return self._fetch_from_api(client_id, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error ingesting CRM data: {str(e)}")
            raise ConnectionError(f"Failed to ingest CRM data: {str(e)}") from e

    def _fetch_from_api(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from CRM API (Salesforce, HubSpot, etc.).
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional API parameters
            
        Returns:
            List of raw CRM records
        """
        lookback_date = self.get_lookback_date()
        
        # Build API parameters based on CRM type
        if self.crm_type == "salesforce":
            return self._fetch_from_salesforce(client_id, lookback_date, **kwargs)
        elif self.crm_type == "hubspot":
            return self._fetch_from_hubspot(client_id, lookback_date, **kwargs)
        else:
            return self._fetch_from_generic_api(client_id, lookback_date, **kwargs)

    def _fetch_from_salesforce(self, client_id: Optional[str], lookback_date: datetime, **kwargs) -> List[Dict[str, Any]]:
        """Fetch data from Salesforce CRM."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # SOQL query for client interactions
        soql_query = f"""
        SELECT Id, AccountId, ContactId, Subject, Description, ActivityDate, Type, 
               Status, Priority, CreatedDate, LastModifiedDate
        FROM Task 
        WHERE ActivityDate >= {lookback_date.strftime('%Y-%m-%d')}
        """
        
        if client_id:
            soql_query += f" AND AccountId = '{client_id}'"
        
        params = {"q": soql_query}
        
        try:
            response = requests.get(
                f"{self.api_base_url}/services/data/v52.0/query",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            records = data.get("records", [])
            
            self.logger.info(f"Fetched {len(records)} CRM records from Salesforce")
            return records
            
        except requests.RequestException as e:
            raise ConnectionError(f"Salesforce API request failed: {str(e)}") from e

    def _fetch_from_database(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from local CRM database.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional database parameters
            
        Returns:
            List of raw CRM records
        """
        if not self.db_connection:
            raise ConnectionError("Database connection not available")
        
        lookback_date = self.get_lookback_date()
        
        # Build SQL query
        query = """
        SELECT 
            id, client_id, interaction_type, interaction_date, 
            description, outcome, priority, status, 
            created_by, created_date, metadata
        FROM crm_interactions 
        WHERE interaction_date >= ?
        """
        
        params = [lookback_date.strftime('%Y-%m-%d')]
        
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        
        # Add additional filters
        if kwargs.get("interaction_type"):
            query += " AND interaction_type = ?"
            params.append(kwargs["interaction_type"])
        
        query += " ORDER BY interaction_date DESC"
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            records = []
            for row in rows:
                record = dict(row)
                # Parse JSON metadata if present
                if record.get("metadata"):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}
                records.append(record)
            
            self.logger.info(f"Fetched {len(records)} CRM records from database")
            return records
            
        except sqlite3.Error as e:
            raise ConnectionError(f"Database query failed: {str(e)}") from e

    def _get_mock_data(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate mock CRM data for testing.
        
        Args:
            client_id: Optional client ID to filter records
            
        Returns:
            List of mock CRM records
        """
        mock_records = [
            {
                "id": "crm_001",
                "client_id": "CLIENT_001",
                "client_name": "Acme Corp",
                "interaction_type": "phone_call",
                "interaction_date": "2025-06-21 14:30:00",
                "description": "Quarterly portfolio review. Client interested in ESG investments.",
                "outcome": "follow_up_scheduled",
                "priority": "high",
                "status": "completed",
                "created_by": "john.doe@bank.com",
                "created_date": "2025-06-21 14:45:00",
                "account_manager": "John Doe",
                "sector": "Technology",
                "aum": 50000000,
                "metadata": {
                    "meeting_duration": 45,
                    "topics_discussed": ["ESG", "portfolio_performance", "market_outlook"],
                    "next_actions": ["send_esg_research", "schedule_followup"]
                }
            },
            {
                "id": "crm_002",
                "client_id": "CLIENT_002",
                "client_name": "Beta Investment Partners",
                "interaction_type": "email",
                "interaction_date": "2025-06-20 09:15:00",
                "description": "Client inquiry about emerging market opportunities",
                "outcome": "information_provided",
                "priority": "medium",
                "status": "completed",
                "created_by": "jane.smith@bank.com",
                "created_date": "2025-06-20 09:30:00",
                "account_manager": "Jane Smith",
                "sector": "Financial Services",
                "aum": 125000000,
                "metadata": {
                    "email_thread_id": "thread_12345",
                    "attachments": ["em_markets_report.pdf"],
                    "regions_of_interest": ["Asia", "Latin America"]
                }
            },
            {
                "id": "crm_003",
                "client_id": "CLIENT_001",
                "client_name": "Acme Corp",
                "interaction_type": "meeting",
                "interaction_date": "2025-06-19 16:00:00",
                "description": "In-person meeting to discuss Q2 earnings impact on portfolio",
                "outcome": "action_items_assigned",
                "priority": "high",
                "status": "completed",
                "created_by": "john.doe@bank.com",
                "created_date": "2025-06-19 17:30:00",
                "account_manager": "John Doe",
                "sector": "Technology",
                "aum": 50000000,
                "metadata": {
                    "meeting_location": "Client Office",
                    "attendees": ["John Doe", "Client CFO", "Client CIO"],
                    "earnings_concerns": ["tech_sector_volatility", "supply_chain_issues"]
                }
            },
            {
                "id": "crm_004",
                "client_id": "CLIENT_003",
                "client_name": "Gamma Pension Fund",
                "interaction_type": "presentation",
                "interaction_date": "2025-06-18 11:00:00",
                "description": "Presented new fixed income strategy to investment committee",
                "outcome": "proposal_under_review",
                "priority": "high",
                "status": "pending",
                "created_by": "mike.wilson@bank.com",
                "created_date": "2025-06-18 12:00:00",
                "account_manager": "Mike Wilson",
                "sector": "Pension/Insurance",
                "aum": 2000000000,
                "metadata": {
                    "presentation_slides": 25,
                    "committee_members": 5,
                    "strategy_focus": "duration_hedging",
                    "decision_timeline": "2_weeks"
                }
            }
        ]
        
        # Filter by client_id if provided
        if client_id:
            mock_records = [record for record in mock_records if record.get("client_id") == client_id]
        
        self.logger.info(f"Generated {len(mock_records)} mock CRM records")
        return mock_records

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw CRM data.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed CRM records
        """
        if not raw_data:
            self.logger.warning("No raw CRM data provided for preprocessing")
            return []
        
        processed_records = []
        
        for record in raw_data:
            try:
                # Clean and standardize the record
                processed_record = self._process_single_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                self.logger.error(f"Error preprocessing CRM record {record.get('id')}: {str(e)}")
        
        self.logger.info(f"Preprocessed {len(processed_records)} CRM records")
        return processed_records

    def _process_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single CRM record."""
        # Standardize field names and clean data
        processed_record = {
            "id": record.get("id") or record.get("Id"),
            "client_id": record.get("client_id") or record.get("AccountId"),
            "client_name": record.get("client_name") or record.get("Account", {}).get("Name"),
            "interaction_type": record.get("interaction_type") or record.get("Type", "").lower(),
            "interaction_date": self._parse_date(record.get("interaction_date") or record.get("ActivityDate")),
            "description": record.get("description") or record.get("Description") or record.get("Subject"),
            "outcome": record.get("outcome") or record.get("Status", "").lower(),
            "priority": record.get("priority") or record.get("Priority", "").lower(),
            "status": record.get("status") or "unknown",
            "created_by": record.get("created_by") or record.get("CreatedBy", {}).get("Email"),
            "created_date": self._parse_date(record.get("created_date") or record.get("CreatedDate")),
            "account_manager": record.get("account_manager"),
            "sector": record.get("sector"),
            "aum": record.get("aum"),
            "metadata": record.get("metadata", {})
        }
        
        # Validate required fields
        if not processed_record["client_id"] or not processed_record["interaction_date"]:
            return None
        
        return processed_record

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            # Common date formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
        
        return None

    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Build features from processed CRM data.
        
        Args:
            processed_data: Preprocessed CRM data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched CRM records
        """
        enriched_records = []
        
        for record in processed_data:
            try:
                enriched_record = record.copy()
                
                # Add derived features
                enriched_record["interaction_recency_days"] = self._calculate_recency(
                    record.get("interaction_date")
                )
                enriched_record["description_length"] = len(record.get("description", ""))
                enriched_record["has_follow_up"] = "follow" in record.get("description", "").lower()
                
                # Extract topics using LLM if available
                if llm_client and record.get("description"):
                    topics = self._extract_topics(record["description"], llm_client)
                    enriched_record["extracted_topics"] = topics
                
                enriched_records.append(enriched_record)
                
            except Exception as e:
                self.logger.error(f"Error building features for CRM record {record.get('id')}: {str(e)}")
                enriched_records.append(record)
        
        return enriched_records

    def _calculate_recency(self, interaction_date: Optional[datetime]) -> Optional[int]:
        """Calculate days since interaction."""
        if not interaction_date:
            return None
        
        return (datetime.now() - interaction_date).days

    def _extract_topics(self, description: str, llm_client) -> List[str]:
        """Extract key topics from interaction description using LLM."""
        try:
            prompt = f"""
            Extract the main topics discussed in this CRM interaction. Focus on investment themes, 
            client concerns, and business opportunities. Return as comma-separated list.
            
            Description: {description}
            
            Topics:
            """
            
            response = llm_client.generate_completion(prompt, max_tokens=100)
            
            if hasattr(response, 'content'):
                response = response.content
            
            # Parse topics
            topics = [topic.strip() for topic in str(response).split(",")]
            return [topic for topic in topics if topic][:5]  # Limit to 5 topics
            
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """
        Test connection to CRM system.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.use_mock_data:
            self.logger.info("Using mock data - connection test passed")
            return True
        
        try:
            if self.database_path and self.db_connection:
                # Test database connection
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                self.logger.info("CRM database connection test passed")
                return True
            
            if self.api_base_url and self.api_key:
                # Test API connection
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(
                    f"{self.api_base_url}/test",
                    headers=headers,
                    timeout=10
                )
                success = response.status_code == 200
                if success:
                    self.logger.info("CRM API connection test passed")
                else:
                    self.logger.error(f"CRM API connection test failed: {response.status_code}")
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"CRM connection test failed: {str(e)}")
            return False

    def get_client_summary(self, client_id: str) -> Dict[str, Any]:
        """
        Get summary information for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client summary information
        """
        try:
            records = self.ingest(client_id=client_id)
            
            if not records:
                return {"client_id": client_id, "interaction_count": 0}
            
            # Calculate summary statistics
            total_interactions = len(records)
            recent_interactions = [r for r in records if 
                                 self._calculate_recency(self._parse_date(r.get("interaction_date"))) <= 30]
            
            interaction_types = {}
            for record in records:
                interaction_type = record.get("interaction_type", "unknown")
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
            
            summary = {
                "client_id": client_id,
                "client_name": records[0].get("client_name"),
                "total_interactions": total_interactions,
                "recent_interactions_30d": len(recent_interactions),
                "interaction_types": interaction_types,
                "last_interaction_date": max([self._parse_date(r.get("interaction_date")) 
                                            for r in records if self._parse_date(r.get("interaction_date"))]),
                "account_manager": records[0].get("account_manager"),
                "sector": records[0].get("sector"),
                "aum": records[0].get("aum")
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting client summary for {client_id}: {str(e)}")
            return {"client_id": client_id, "error": str(e)} 