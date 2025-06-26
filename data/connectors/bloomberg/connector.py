"""
Bloomberg Chat Connector Implementation.
This module handles Bloomberg chat data ingestion, preprocessing, and feature extraction.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from ..base_connector import BaseConnector
from ..schemas import ChatMessage, ConnectorConfig
from .preprocessor import BloombergChatPreprocessor
from .feature_builder import BloombergChatFeatureBuilder


class BloombergChatConnector(BaseConnector):
    """
    Connector for Bloomberg Chat data.
    
    Handles ingestion from Bloomberg API, preprocessing of chat messages,
    and LLM-powered feature extraction including summaries, keywords, and sentiment.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bloomberg Chat connector.
        
        Args:
            config: Configuration dictionary containing Bloomberg API settings
        """
        # Set default connector name if not provided
        if "connector_name" not in config:
            config["connector_name"] = "bloomberg_chat"
            
        super().__init__(config)
        
        # Initialize preprocessor and feature builder
        self.preprocessor = BloombergChatPreprocessor()
        self.feature_builder = BloombergChatFeatureBuilder()
        
        # Bloomberg API configuration
        self.api_base_url = config.get("api_base_url", os.getenv("BLOOMBERG_API_URL"))
        self.api_key = config.get("api_key", os.getenv("BLOOMBERG_API_KEY"))
        self.timeout = config.get("timeout", 30)
        
        if not self.api_base_url or not self.api_key:
            self.logger.warning("Bloomberg API credentials not found. Using mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False

    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest Bloomberg chat data.
        
        Args:
            client_id: Optional client ID to filter conversations
            **kwargs: Additional parameters (lookback_hours, conversation_id, etc.)
            
        Returns:
            List of raw chat message records
            
        Raises:
            ConnectionError: If unable to connect to Bloomberg API
            ValueError: If invalid parameters provided
        """
        try:
            if self.use_mock_data:
                return self._get_mock_data(client_id)
            
            return self._fetch_from_api(client_id, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error ingesting Bloomberg chat data: {str(e)}")
            raise ConnectionError(f"Failed to ingest Bloomberg chat data: {str(e)}") from e

    def _fetch_from_api(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from Bloomberg API.
        
        Args:
            client_id: Optional client ID to filter conversations
            **kwargs: Additional API parameters
            
        Returns:
            List of raw chat records
        """
        lookback_date = self.get_lookback_date()
        
        # Build API parameters
        params = {
            "start_date": lookback_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "format": "json"
        }
        
        if client_id:
            params["client_id"] = client_id
            
        # Add additional parameters
        params.update(kwargs)
        
        # API request headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RecommendationSystem/1.0"
        }
        
        try:
            response = requests.get(
                f"{self.api_base_url}/chat/messages",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            messages = data.get("messages", [])
            
            self.logger.info(f"Fetched {len(messages)} chat messages from Bloomberg API")
            return messages
            
        except requests.RequestException as e:
            raise ConnectionError(f"Bloomberg API request failed: {str(e)}") from e

    def _get_mock_data(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate mock Bloomberg chat data for testing.
        
        Args:
            client_id: Optional client ID to filter conversations
            
        Returns:
            List of mock chat records
        """
        mock_messages = [
            {
                "message_id": "bbg_msg_001",
                "timestamp": "2025-06-20 10:15:00",
                "participants": ["Alex", "Chen"],
                "message_content": "We're really seeing pressure on our shipping costs, any ideas on tech solutions?",
                "client_id": "Chen",
                "conversation_id": "conv_001",
                "message_type": "text",
                "channel": "bloomberg_chat"
            },
            {
                "message_id": "bbg_msg_002", 
                "timestamp": "2025-06-20 10:16:30",
                "participants": ["Chen", "Alex"],
                "message_content": "Looking for automation solutions specifically for logistics. Any research on supply chain tech?",
                "client_id": "Chen",
                "conversation_id": "conv_001",
                "message_type": "text",
                "channel": "bloomberg_chat"
            },
            {
                "message_id": "bbg_msg_003",
                "timestamp": "2025-06-18 11:00:00",
                "participants": ["Alex", "Maria"],
                "message_content": "Client mentioned interest in our tech sector research during quarterly review.",
                "client_id": "Maria",
                "conversation_id": "conv_002",
                "message_type": "text",
                "channel": "bloomberg_chat"
            },
            {
                "message_id": "bbg_msg_004",
                "timestamp": "2025-06-19 14:30:00",
                "participants": ["Chen", "Alex"],
                "message_content": "USD/CAD rates looking volatile. Need analysis on currency trends.",
                "client_id": "Chen",
                "conversation_id": "conv_003",
                "message_type": "text",
                "channel": "bloomberg_chat"
            }
        ]
        
        # Filter by client_id if provided
        if client_id:
            mock_messages = [msg for msg in mock_messages if msg.get("client_id") == client_id]
        
        self.logger.info(f"Generated {len(mock_messages)} mock Bloomberg chat messages")
        return mock_messages

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw Bloomberg chat data.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed chat message records
        """
        try:
            return self.preprocessor.process(raw_data)
        except Exception as e:
            self.logger.error(f"Error preprocessing Bloomberg chat data: {str(e)}")
            raise ValueError(f"Bloomberg chat preprocessing failed: {str(e)}") from e

    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Build features from processed Bloomberg chat data using LLM.
        
        Args:
            processed_data: Preprocessed chat data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched chat records
        """
        try:
            return self.feature_builder.build_features(processed_data, llm_client)
        except Exception as e:
            self.logger.error(f"Error building Bloomberg chat features: {str(e)}")
            raise RuntimeError(f"Bloomberg chat feature building failed: {str(e)}") from e

    def test_connection(self) -> bool:
        """
        Test connection to Bloomberg API.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.use_mock_data:
            self.logger.info("Using mock data - connection test passed")
            return True
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.api_base_url}/health",
                headers=headers,
                timeout=10
            )
            
            success = response.status_code == 200
            if success:
                self.logger.info("Bloomberg API connection test passed")
            else:
                self.logger.error(f"Bloomberg API connection test failed: {response.status_code}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Bloomberg API connection test failed: {str(e)}")
            return False

    def get_conversation_summary(self, conversation_id: str, 
                               llm_client=None) -> Optional[str]:
        """
        Get summary of a specific conversation.
        
        Args:
            conversation_id: Conversation ID to summarize
            llm_client: LLM client for summarization
            
        Returns:
            Conversation summary or None if not found
        """
        try:
            # Fetch messages for specific conversation
            raw_data = self.ingest(conversation_id=conversation_id)
            if not raw_data:
                return None
                
            # Process and build features
            processed_data = self.preprocess(raw_data)
            features = self.build_features(processed_data, llm_client)
            
            # Combine all messages for summary
            all_messages = [msg.get("message_content", "") for msg in features]
            conversation_text = "\n".join(all_messages)
            
            if llm_client:
                prompt = f"Summarize this Bloomberg chat conversation:\n\n{conversation_text}"
                summary = llm_client.generate_completion(prompt, max_tokens=200)
                return summary
            else:
                return f"Conversation contains {len(all_messages)} messages"
                
        except Exception as e:
            self.logger.error(f"Error getting conversation summary: {str(e)}")
            return None

    def get_client_interaction_history(self, client_id: str, 
                                     days: int = 30) -> List[Dict[str, Any]]:
        """
        Get interaction history for a specific client.
        
        Args:
            client_id: Client ID to get history for
            days: Number of days to look back
            
        Returns:
            List of client interactions
        """
        try:
            # Temporarily override lookback period
            original_lookback = self.config.lookback_days
            self.config.lookback_days = days
            
            # Fetch and process data
            raw_data = self.ingest(client_id=client_id)
            processed_data = self.preprocess(raw_data)
            
            # Restore original lookback period
            self.config.lookback_days = original_lookback
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error getting client interaction history: {str(e)}")
            return [] 