"""
Readership Connector Implementation.
This module handles readership data ingestion for research consumption patterns and engagement metrics.
"""

import os
import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .base_connector import BaseConnector
from .schemas import ReadearshipData, ConnectorConfig


class ReadershipConnector(BaseConnector):
    """
    Connector for readership data including research consumption patterns, engagement metrics, and preferences.
    
    Tracks how clients interact with research publications to inform recommendation relevance.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Readership connector.
        
        Args:
            config: Configuration dictionary containing readership settings
        """
        # Set default connector name if not provided
        if "connector_name" not in config:
            config["connector_name"] = "readership"
            
        super().__init__(config)
        
        # Readership system configuration
        self.analytics_url = config.get("analytics_url", os.getenv("ANALYTICS_URL"))
        self.api_key = config.get("api_key", os.getenv("ANALYTICS_API_KEY"))
        self.database_path = config.get("database_path", os.getenv("READERSHIP_DATABASE_PATH"))
        self.log_files_path = config.get("log_files_path", os.getenv("LOG_FILES_PATH"))
        self.timeout = config.get("timeout", 30)
        
        # Database connection for local readership data
        self.db_connection = None
        
        if not any([self.analytics_url, self.database_path, self.log_files_path]):
            self.logger.warning("No readership data source configured. Using mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            
        if self.database_path:
            self._init_database()

    def _init_database(self):
        """Initialize database connection for local readership data."""
        try:
            self.db_connection = sqlite3.connect(self.database_path)
            self.db_connection.row_factory = sqlite3.Row
            self.logger.info(f"Connected to readership database: {self.database_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to readership database: {str(e)}")
            self.use_mock_data = True

    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest readership data.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional parameters (publication_type, engagement_type, etc.)
            
        Returns:
            List of raw readership records
            
        Raises:
            ConnectionError: If unable to connect to analytics system
            ValueError: If invalid parameters provided
        """
        try:
            if self.use_mock_data:
                return self._get_mock_data(client_id)
            
            if self.database_path:
                return self._fetch_from_database(client_id, **kwargs)
            elif self.log_files_path:
                return self._fetch_from_logs(client_id, **kwargs)
            else:
                return self._fetch_from_api(client_id, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error ingesting readership data: {str(e)}")
            raise ConnectionError(f"Failed to ingest readership data: {str(e)}") from e

    def _fetch_from_api(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from analytics API.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional API parameters
            
        Returns:
            List of raw readership records
        """
        lookback_date = self.get_lookback_date()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "start_date": lookback_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "event_types": ["view", "download", "share", "bookmark", "time_spent"],
            "include_metadata": True
        }
        
        if client_id:
            params["client_id"] = client_id
        
        try:
            response = requests.get(
                f"{self.analytics_url}/api/v1/readership",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            records = data.get("events", [])
            
            self.logger.info(f"Fetched {len(records)} readership records from analytics API")
            return records
            
        except requests.RequestException as e:
            raise ConnectionError(f"Analytics API request failed: {str(e)}") from e

    def _fetch_from_database(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from local readership database.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional database parameters
            
        Returns:
            List of raw readership records
        """
        if not self.db_connection:
            raise ConnectionError("Database connection not available")
        
        lookback_date = self.get_lookback_date()
        
        query = """
        SELECT 
            event_id, client_id, publication_id, publication_title,
            event_type, event_timestamp, time_spent_seconds,
            page_views, scroll_depth, download_count,
            share_count, bookmark_count, rating, 
            user_agent, ip_address, referrer, metadata
        FROM readership_events 
        WHERE event_timestamp >= ?
        """
        
        params = [lookback_date.strftime('%Y-%m-%d %H:%M:%S')]
        
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        
        if kwargs.get("publication_type"):
            query += " AND publication_id LIKE ?"
            params.append(f"%{kwargs['publication_type']}%")
        
        query += " ORDER BY event_timestamp DESC"
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record = dict(row)
                if record.get("metadata"):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        record["metadata"] = {}
                records.append(record)
            
            self.logger.info(f"Fetched {len(records)} readership records from database")
            return records
            
        except sqlite3.Error as e:
            raise ConnectionError(f"Database query failed: {str(e)}") from e

    def _get_mock_data(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate mock readership data for testing.
        
        Args:
            client_id: Optional client ID to filter records
            
        Returns:
            List of mock readership records
        """
        mock_records = [
            {
                "event_id": "read_001",
                "client_id": "CLIENT_001",
                "client_name": "Acme Corp",
                "publication_id": "PUB_2025_001",
                "publication_title": "Q2 2025 Technology Sector Outlook",
                "publication_type": "sector_report",
                "author": "Jane Smith",
                "event_type": "view",
                "event_timestamp": "2025-06-21 09:30:00",
                "time_spent_seconds": 450,
                "page_views": 8,
                "scroll_depth": 0.85,
                "download_count": 1,
                "share_count": 0,
                "bookmark_count": 1,
                "rating": 4,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "referrer": "email_campaign",
                "metadata": {
                    "device_type": "desktop",
                    "session_id": "sess_12345",
                    "campaign_id": "tech_sector_2025",
                    "sections_viewed": ["executive_summary", "key_themes", "stock_picks"]
                }
            },
            {
                "event_id": "read_002",
                "client_id": "CLIENT_002",
                "client_name": "Beta Investment Partners",
                "publication_id": "PUB_2025_002",
                "publication_title": "Emerging Markets FX Strategy",
                "publication_type": "strategy_note",
                "author": "Mike Wilson",
                "event_type": "download",
                "event_timestamp": "2025-06-21 11:15:00",
                "time_spent_seconds": 0,
                "page_views": 0,
                "scroll_depth": 0,
                "download_count": 1,
                "share_count": 2,
                "bookmark_count": 0,
                "rating": None,
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "referrer": "direct",
                "metadata": {
                    "device_type": "desktop",
                    "download_format": "pdf",
                    "shared_channels": ["email", "bloomberg_chat"]
                }
            },
            {
                "event_id": "read_003",
                "client_id": "CLIENT_001",
                "client_name": "Acme Corp",
                "publication_id": "PUB_2025_003",
                "publication_title": "ESG Investment Trends 2025",
                "publication_type": "thematic_report",
                "author": "Sarah Johnson",
                "event_type": "view",
                "event_timestamp": "2025-06-20 16:45:00",
                "time_spent_seconds": 720,
                "page_views": 12,
                "scroll_depth": 0.95,
                "download_count": 1,
                "share_count": 1,
                "bookmark_count": 1,
                "rating": 5,
                "user_agent": "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X)",
                "referrer": "research_portal",
                "metadata": {
                    "device_type": "tablet",
                    "session_id": "sess_67890",
                    "high_engagement": True,
                    "sections_viewed": ["introduction", "esg_trends", "investment_implications", "portfolio_construction"]
                }
            },
            {
                "event_id": "read_004",
                "client_id": "CLIENT_003",
                "client_name": "Gamma Pension Fund",
                "publication_id": "PUB_2025_004",
                "publication_title": "Fixed Income Market Weekly",
                "publication_type": "weekly_update",
                "author": "David Lee",
                "event_type": "view",
                "event_timestamp": "2025-06-19 08:30:00",
                "time_spent_seconds": 180,
                "page_views": 3,
                "scroll_depth": 0.60,
                "download_count": 0,
                "share_count": 0,
                "bookmark_count": 0,
                "rating": None,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "referrer": "email_newsletter",
                "metadata": {
                    "device_type": "desktop",
                    "session_id": "sess_11111",
                    "quick_scan": True,
                    "exit_page": "page_3"
                }
            },
            {
                "event_id": "read_005",
                "client_id": "CLIENT_002",
                "client_name": "Beta Investment Partners",
                "publication_id": "PUB_2025_001",
                "publication_title": "Q2 2025 Technology Sector Outlook",
                "publication_type": "sector_report",
                "author": "Jane Smith",
                "event_type": "share",
                "event_timestamp": "2025-06-18 14:20:00",
                "time_spent_seconds": 0,
                "page_views": 0,
                "scroll_depth": 0,
                "download_count": 0,
                "share_count": 1,
                "bookmark_count": 0,
                "rating": None,
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X)",
                "referrer": "mobile_app",
                "metadata": {
                    "device_type": "mobile",
                    "share_channel": "email",
                    "share_recipients": 3,
                    "share_note": "Relevant for our tech allocation review"
                }
            }
        ]
        
        # Filter by client_id if provided
        if client_id:
            mock_records = [record for record in mock_records if record.get("client_id") == client_id]
        
        self.logger.info(f"Generated {len(mock_records)} mock readership records")
        return mock_records

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw readership data.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed readership records
        """
        if not raw_data:
            self.logger.warning("No raw readership data provided for preprocessing")
            return []
        
        processed_records = []
        
        for record in raw_data:
            try:
                processed_record = self._process_single_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                self.logger.error(f"Error preprocessing readership record {record.get('event_id')}: {str(e)}")
        
        self.logger.info(f"Preprocessed {len(processed_records)} readership records")
        return processed_records

    def _process_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single readership record."""
        processed_record = {
            "event_id": record.get("event_id"),
            "client_id": record.get("client_id"),
            "client_name": record.get("client_name"),
            "publication_id": record.get("publication_id"),
            "publication_title": record.get("publication_title"),
            "publication_type": record.get("publication_type", "").lower(),
            "author": record.get("author"),
            "event_type": record.get("event_type", "").lower(),
            "event_timestamp": self._parse_timestamp(record.get("event_timestamp")),
            "time_spent_seconds": record.get("time_spent_seconds", 0),
            "page_views": record.get("page_views", 0),
            "scroll_depth": record.get("scroll_depth", 0.0),
            "download_count": record.get("download_count", 0),
            "share_count": record.get("share_count", 0),
            "bookmark_count": record.get("bookmark_count", 0),
            "rating": record.get("rating"),
            "user_agent": record.get("user_agent"),
            "referrer": record.get("referrer"),
            "metadata": record.get("metadata", {})
        }
        
        # Validate required fields
        if not processed_record["event_id"] or not processed_record["client_id"]:
            return None
        
        if not processed_record["event_timestamp"]:
            return None
        
        # Normalize event types
        valid_event_types = ["view", "download", "share", "bookmark", "rating"]
        if processed_record["event_type"] not in valid_event_types:
            processed_record["event_type"] = "view"
        
        return processed_record

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not timestamp:
            return None
        
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
        
        return None

    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Build features from processed readership data.
        
        Args:
            processed_data: Preprocessed readership data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched readership records
        """
        enriched_records = []
        
        for record in processed_data:
            try:
                enriched_record = record.copy()
                
                # Add engagement features
                enriched_record.update(self._calculate_engagement_features(record))
                
                # Extract content preferences using LLM if available
                if llm_client and record.get("publication_title"):
                    content_themes = self._extract_content_themes(record, llm_client)
                    enriched_record["content_themes"] = content_themes
                
                enriched_records.append(enriched_record)
                
            except Exception as e:
                self.logger.error(f"Error building features for readership {record.get('event_id')}: {str(e)}")
                enriched_records.append(record)
        
        return enriched_records

    def _calculate_engagement_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate engagement-specific features."""
        features = {}
        
        # Engagement scoring
        time_spent = record.get("time_spent_seconds", 0)
        page_views = record.get("page_views", 0)
        scroll_depth = record.get("scroll_depth", 0.0)
        
        # High engagement thresholds
        features["is_high_engagement"] = (
            time_spent > 300 and  # 5+ minutes
            scroll_depth > 0.7 and  # 70%+ scroll
            page_views > 5
        )
        
        # Engagement score (0-10)
        engagement_score = 0
        if time_spent > 0:
            engagement_score += min(time_spent / 60, 5)  # Up to 5 points for time
        if scroll_depth > 0:
            engagement_score += scroll_depth * 3  # Up to 3 points for scroll
        if page_views > 0:
            engagement_score += min(page_views / 5, 2)  # Up to 2 points for page views
        
        features["engagement_score"] = round(engagement_score, 2)
        
        # Content interaction features
        features["downloaded"] = record.get("download_count", 0) > 0
        features["shared"] = record.get("share_count", 0) > 0
        features["bookmarked"] = record.get("bookmark_count", 0) > 0
        features["rated"] = record.get("rating") is not None
        
        # Device categorization
        user_agent = record.get("user_agent", "").lower()
        if "mobile" in user_agent or "iphone" in user_agent:
            features["device_category"] = "mobile"
        elif "tablet" in user_agent or "ipad" in user_agent:
            features["device_category"] = "tablet"
        else:
            features["device_category"] = "desktop"
        
        # Traffic source categorization
        referrer = record.get("referrer", "").lower()
        if "email" in referrer:
            features["traffic_source"] = "email"
        elif "search" in referrer:
            features["traffic_source"] = "search"
        elif "direct" in referrer:
            features["traffic_source"] = "direct"
        elif "social" in referrer:
            features["traffic_source"] = "social"
        else:
            features["traffic_source"] = "other"
        
        return features

    def _extract_content_themes(self, record: Dict[str, Any], llm_client) -> List[str]:
        """Extract content themes using LLM."""
        try:
            prompt = f"""
            Analyze this research publication title and extract the main investment themes.
            Focus on sectors, asset classes, strategies, and market conditions.
            
            Title: {record.get('publication_title')}
            Type: {record.get('publication_type')}
            Author: {record.get('author')}
            
            Return top 3 investment themes as comma-separated list:
            """
            
            response = llm_client.generate_completion(prompt, max_tokens=50)
            
            if hasattr(response, 'content'):
                response = response.content
            
            # Parse themes
            themes = [theme.strip() for theme in str(response).split(",")]
            return [theme for theme in themes if theme][:3]
            
        except Exception as e:
            self.logger.error(f"Error extracting content themes: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """
        Test connection to readership system.
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.use_mock_data:
            self.logger.info("Using mock data - connection test passed")
            return True
        
        try:
            if self.database_path and self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                self.logger.info("Readership database connection test passed")
                return True
            
            if self.analytics_url and self.api_key:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(
                    f"{self.analytics_url}/api/v1/health",
                    headers=headers,
                    timeout=10
                )
                success = response.status_code == 200
                if success:
                    self.logger.info("Analytics API connection test passed")
                else:
                    self.logger.error(f"Analytics API connection test failed: {response.status_code}")
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"Readership connection test failed: {str(e)}")
            return False

    def get_client_reading_profile(self, client_id: str) -> Dict[str, Any]:
        """
        Get reading profile for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client reading profile with preferences and engagement patterns
        """
        try:
            records = self.ingest(client_id=client_id)
            processed_records = self.preprocess(records)
            
            if not processed_records:
                return {"client_id": client_id, "total_interactions": 0}
            
            # Calculate reading statistics
            total_interactions = len(processed_records)
            
            # Content type preferences
            content_types = {}
            for record in processed_records:
                pub_type = record.get("publication_type", "unknown")
                content_types[pub_type] = content_types.get(pub_type, 0) + 1
            
            # Author preferences
            authors = {}
            for record in processed_records:
                author = record.get("author", "unknown")
                authors[author] = authors.get(author, 0) + 1
            
            # Engagement metrics
            high_engagement_count = sum(1 for r in processed_records 
                                      if r.get("time_spent_seconds", 0) > 300)
            downloads = sum(r.get("download_count", 0) for r in processed_records)
            shares = sum(r.get("share_count", 0) for r in processed_records)
            bookmarks = sum(r.get("bookmark_count", 0) for r in processed_records)
            
            # Average engagement
            avg_time_spent = sum(r.get("time_spent_seconds", 0) for r in processed_records) / total_interactions
            avg_scroll_depth = sum(r.get("scroll_depth", 0) for r in processed_records) / total_interactions
            
            profile = {
                "client_id": client_id,
                "total_interactions": total_interactions,
                "high_engagement_rate": round(high_engagement_count / total_interactions, 3) if total_interactions > 0 else 0,
                "preferred_content_types": dict(sorted(content_types.items(), key=lambda x: x[1], reverse=True)),
                "preferred_authors": dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]),
                "total_downloads": downloads,
                "total_shares": shares,
                "total_bookmarks": bookmarks,
                "avg_time_spent_seconds": round(avg_time_spent, 1),
                "avg_scroll_depth": round(avg_scroll_depth, 3),
                "last_interaction_date": max([r.get("event_timestamp") for r in processed_records 
                                            if r.get("event_timestamp")]) if processed_records else None
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting reading profile for {client_id}: {str(e)}")
            return {"client_id": client_id, "error": str(e)} 