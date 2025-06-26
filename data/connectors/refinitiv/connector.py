"""
Refinitiv Chat Connector Implementation.
This module handles Refinitiv chat data ingestion, preprocessing, and feature extraction.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from ..base_connector import BaseConnector
from ..schemas import ChatMessage, ConnectorConfig
from .preprocessor import RefinitivChatPreprocessor
from .feature_builder import RefinitivChatFeatureBuilder


class RefinitivChatConnector(BaseConnector):
    """
    Connector for Refinitiv Chat data (formerly Thomson Reuters).
    
    Handles ingestion from Refinitiv Eikon/Workspace API, preprocessing of chat messages,
    and LLM-powered feature extraction including summaries, keywords, and sentiment.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Refinitiv Chat connector.
        
        Args:
            config: Configuration dictionary containing Refinitiv API settings
        """
        # Set default connector name if not provided
        if "connector_name" not in config:
            config["connector_name"] = "refinitiv_chat"
            
        super().__init__(config)
        
        # Initialize preprocessor and feature builder
        self.preprocessor = RefinitivChatPreprocessor()
        self.feature_builder = RefinitivChatFeatureBuilder()
        
        # Refinitiv API configuration
        self.api_base_url = config.get("api_base_url", os.getenv("REFINITIV_API_URL"))
        self.api_key = config.get("api_key", os.getenv("REFINITIV_API_KEY"))
        self.app_key = config.get("app_key", os.getenv("REFINITIV_APP_KEY"))
        self.timeout = config.get("timeout", 30)
        
        if not self.api_base_url or not self.api_key or not self.app_key:
            self.logger.warning("Refinitiv API credentials not found. Using mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False

    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest Refinitiv chat data.
        
        Args:
            client_id: Optional client ID to filter conversations
            **kwargs: Additional parameters (lookback_hours, conversation_id, etc.)
            
        Returns:
            List of raw chat message records
            
        Raises:
            ConnectionError: If unable to connect to Refinitiv API
            ValueError: If invalid parameters provided
        """
        try:
            if self.use_mock_data:
                return self._get_mock_data(client_id)
            
            return self._fetch_from_api(client_id, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error ingesting Refinitiv chat data: {str(e)}")
            raise ConnectionError(f"Failed to ingest Refinitiv chat data: {str(e)}") from e

    def _fetch_from_api(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from Refinitiv API.
        
        Args:
            client_id: Optional client ID to filter conversations
            **kwargs: Additional API parameters
            
        Returns:
            List of raw chat records
        """
        lookback_date = self.get_lookback_date()
        
        # Build API parameters
        params = {
            "startDate": lookback_date.isoformat(),
            "endDate": datetime.now().isoformat(),
            "format": "json"
        }
        
        if client_id:
            params["clientId"] = client_id
            
        # Add additional parameters
        params.update(kwargs)
        
        # API request headers (Refinitiv uses different auth)
        headers = {
            "Authorization": f"Token {self.api_key}",
            "X-TR-ApplicationID": self.app_key,
            "Content-Type": "application/json",
            "User-Agent": "RecommendationSystem/1.0"
        }
        
        try:
            response = requests.get(
                f"{self.api_base_url}/messenger/v1/messages",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            messages = data.get("data", {}).get("messages", [])
            
            self.logger.info(f"Fetched {len(messages)} chat messages from Refinitiv API")
            return messages
            
        except requests.RequestException as e:
            raise ConnectionError(f"Refinitiv API request failed: {str(e)}") from e

    def _get_mock_data(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate mock Refinitiv chat data for testing.
        
        Args:
            client_id: Optional client ID to filter conversations
            
        Returns:
            List of mock chat records
        """
        mock_messages = [
            {
                "message_id": "rf_msg_001",
                "timestamp": "2025-06-21 09:30:00",
                "participants": ["Alex", "Chen"],
                "message_content": "What's your view on EUR/USD with the ECB meeting coming up?",
                "client_id": "Chen",
                "conversation_id": "rf_conv_001",
                "message_type": "text",
                "channel": "refinitiv_chat",
                "room_name": "FX Trading Room"
            },
            {
                "message_id": "rf_msg_002", 
                "timestamp": "2025-06-21 09:31:15",
                "participants": ["Chen", "Alex"],
                "message_content": "Expecting dovish tone. Looking at downside targets around 1.0850",
                "client_id": "Chen",
                "conversation_id": "rf_conv_001",
                "message_type": "text",
                "channel": "refinitiv_chat",
                "room_name": "FX Trading Room"
            },
            {
                "message_id": "rf_msg_003",
                "timestamp": "2025-06-19 15:45:00",
                "participants": ["Maria", "Alex"],
                "message_content": "Any research on emerging market tech companies? Client interested in Vietnam market.",
                "client_id": "Maria",
                "conversation_id": "rf_conv_002",
                "message_type": "text",
                "channel": "refinitiv_chat",
                "room_name": "APAC Research"
            },
            {
                "message_id": "rf_msg_004",
                "timestamp": "2025-06-22 11:20:00",
                "participants": ["Chen", "Alex"],
                "message_content": "Oil inventories data disappointing. WTI looking weak below $70",
                "client_id": "Chen",
                "conversation_id": "rf_conv_003",
                "message_type": "text",
                "channel": "refinitiv_chat",
                "room_name": "Commodities Desk"
            }
        ]
        
        # Filter by client_id if provided
        if client_id:
            mock_messages = [msg for msg in mock_messages if msg.get("client_id") == client_id]
        
        self.logger.info(f"Generated {len(mock_messages)} mock Refinitiv chat messages")
        return mock_messages

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw Refinitiv chat data.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed chat message records
        """
        try:
            return self.preprocessor.process(raw_data)
        except Exception as e:
            self.logger.error(f"Error preprocessing Refinitiv chat data: {str(e)}")
            raise ValueError(f"Refinitiv chat preprocessing failed: {str(e)}") from e

    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Build features from processed Refinitiv chat data using LLM.
        
        Args:
            processed_data: Preprocessed chat data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched chat records
        """
        try:
            return self.feature_builder.build_features(processed_data, llm_client)
        except Exception as e:
            self.logger.error(f"Error building Refinitiv chat features: {str(e)}")
            raise RuntimeError(f"Refinitiv chat feature building failed: {str(e)}") from e

    def test_connection(self) -> bool:
        """
        Test connection to Refinitiv API.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.use_mock_data:
            self.logger.info("Using mock data - connection test passed")
            return True
            
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "X-TR-ApplicationID": self.app_key,
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_base_url}/common/v1/ping",
                headers=headers,
                timeout=10
            )
            
            success = response.status_code == 200
            if success:
                self.logger.info("Refinitiv API connection test passed")
            else:
                self.logger.error(f"Refinitiv API connection test failed: {response.status_code}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Refinitiv API connection test failed: {str(e)}")
            return False

    def get_available_rooms(self) -> List[Dict[str, Any]]:
        """
        Get list of available chat rooms.
        
        Returns:
            List of available rooms
        """
        try:
            if self.use_mock_data:
                return [
                    {"room_id": "fx_trading", "name": "FX Trading Room", "description": "Foreign Exchange discussions"},
                    {"room_id": "apac_research", "name": "APAC Research", "description": "Asia Pacific research and insights"},
                    {"room_id": "commodities", "name": "Commodities Desk", "description": "Commodities trading and analysis"}
                ]
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "X-TR-ApplicationID": self.app_key,
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_base_url}/messenger/v1/rooms",
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", {}).get("rooms", [])
            
        except Exception as e:
            self.logger.error(f"Error getting available rooms: {str(e)}")
            return [] 