"""
KYC (Know Your Customer) Connector Implementation.
This module handles KYC data ingestion for client profiles, compliance data, and investment preferences.
"""

import os
import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from .base_connector import BaseConnector
from .schemas import ClientProfile, ConnectorConfig


class KYCConnector(BaseConnector):
    """
    Connector for KYC data including client profiles, investment preferences, and compliance information.
    
    Provides comprehensive client profiling for personalized research recommendations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KYC connector.
        
        Args:
            config: Configuration dictionary containing KYC settings
        """
        # Set default connector name if not provided
        if "connector_name" not in config:
            config["connector_name"] = "kyc"
            
        super().__init__(config)
        
        # KYC system configuration
        self.kyc_system_url = config.get("kyc_system_url", os.getenv("KYC_SYSTEM_URL"))
        self.api_key = config.get("api_key", os.getenv("KYC_API_KEY"))
        self.database_path = config.get("database_path", os.getenv("KYC_DATABASE_PATH"))
        self.timeout = config.get("timeout", 30)
        
        # Database connection for local KYC data
        self.db_connection = None
        
        if not self.kyc_system_url and not self.database_path:
            self.logger.warning("No KYC data source configured. Using mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            
        if self.database_path:
            self._init_database()

    def _init_database(self):
        """Initialize database connection for local KYC data."""
        try:
            self.db_connection = sqlite3.connect(self.database_path)
            self.db_connection.row_factory = sqlite3.Row
            self.logger.info(f"Connected to KYC database: {self.database_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to KYC database: {str(e)}")
            self.use_mock_data = True

    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest KYC data.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional parameters (include_inactive, profile_type, etc.)
            
        Returns:
            List of raw KYC records
            
        Raises:
            ConnectionError: If unable to connect to KYC system
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
            self.logger.error(f"Error ingesting KYC data: {str(e)}")
            raise ConnectionError(f"Failed to ingest KYC data: {str(e)}") from e

    def _fetch_from_api(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from KYC system API.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional API parameters
            
        Returns:
            List of raw KYC records
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "include_inactive": kwargs.get("include_inactive", False),
            "include_compliance": True,
            "include_preferences": True
        }
        
        if client_id:
            params["client_id"] = client_id
        
        try:
            response = requests.get(
                f"{self.kyc_system_url}/api/v1/profiles",
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            records = data.get("profiles", [])
            
            self.logger.info(f"Fetched {len(records)} KYC records from API")
            return records
            
        except requests.RequestException as e:
            raise ConnectionError(f"KYC API request failed: {str(e)}") from e

    def _fetch_from_database(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from local KYC database.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional database parameters
            
        Returns:
            List of raw KYC records
        """
        if not self.db_connection:
            raise ConnectionError("Database connection not available")
        
        query = """
        SELECT 
            client_id, client_name, client_type, incorporation_country,
            business_sector, aum, risk_tolerance, investment_objectives,
            geographic_preferences, asset_class_preferences,
            esg_preferences, compliance_status, kyc_last_updated,
            account_manager, metadata
        FROM kyc_profiles
        WHERE compliance_status = 'active'
        """
        
        params = []
        
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        
        if not kwargs.get("include_inactive", False):
            query += " AND compliance_status != 'inactive'"
        
        query += " ORDER BY kyc_last_updated DESC"
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                record = dict(row)
                # Parse JSON fields
                for json_field in ["investment_objectives", "geographic_preferences", 
                                 "asset_class_preferences", "esg_preferences", "metadata"]:
                    if record.get(json_field):
                        try:
                            record[json_field] = json.loads(record[json_field])
                        except json.JSONDecodeError:
                            record[json_field] = {}
                records.append(record)
            
            self.logger.info(f"Fetched {len(records)} KYC records from database")
            return records
            
        except sqlite3.Error as e:
            raise ConnectionError(f"Database query failed: {str(e)}") from e

    def _get_mock_data(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate mock KYC data for testing.
        
        Args:
            client_id: Optional client ID to filter records
            
        Returns:
            List of mock KYC records
        """
        mock_records = [
            {
                "client_id": "CLIENT_001",
                "client_name": "Acme Corp",
                "client_type": "corporate",
                "incorporation_country": "United States",
                "business_sector": "Technology",
                "aum": 50000000,
                "risk_tolerance": "moderate",
                "investment_objectives": {
                    "primary": "growth",
                    "secondary": "income",
                    "time_horizon": "long_term",
                    "return_target": 8.5,
                    "volatility_tolerance": "medium"
                },
                "geographic_preferences": {
                    "home_bias": 60,
                    "developed_markets": 30,
                    "emerging_markets": 10,
                    "preferred_regions": ["North America", "Europe"],
                    "restricted_countries": ["Russia", "Iran"]
                },
                "asset_class_preferences": {
                    "equities": 65,
                    "fixed_income": 25,
                    "alternatives": 10,
                    "cash": 0,
                    "preferred_sectors": ["Technology", "Healthcare", "Financial Services"],
                    "restricted_sectors": ["Tobacco", "Weapons"]
                },
                "esg_preferences": {
                    "esg_integration": True,
                    "exclusion_screening": True,
                    "impact_investing": False,
                    "climate_focus": True,
                    "esg_minimum_rating": "B",
                    "carbon_footprint_limit": True
                },
                "compliance_status": "active",
                "kyc_last_updated": "2025-03-15",
                "account_manager": "john.doe@bank.com",
                "metadata": {
                    "onboarding_date": "2020-01-15",
                    "last_review_date": "2025-03-15",
                    "next_review_date": "2026-03-15",
                    "suitability_score": 8.2,
                    "complexity_level": "sophisticated"
                }
            },
            {
                "client_id": "CLIENT_002",
                "client_name": "Beta Investment Partners",
                "client_type": "institutional",
                "incorporation_country": "United Kingdom",
                "business_sector": "Financial Services",
                "aum": 125000000,
                "risk_tolerance": "aggressive",
                "investment_objectives": {
                    "primary": "total_return",
                    "secondary": "diversification",
                    "time_horizon": "medium_term",
                    "return_target": 12.0,
                    "volatility_tolerance": "high"
                },
                "geographic_preferences": {
                    "home_bias": 40,
                    "developed_markets": 35,
                    "emerging_markets": 25,
                    "preferred_regions": ["Europe", "Asia Pacific", "North America"],
                    "restricted_countries": []
                },
                "asset_class_preferences": {
                    "equities": 50,
                    "fixed_income": 20,
                    "alternatives": 25,
                    "cash": 5,
                    "preferred_sectors": ["Technology", "Consumer Discretionary", "Industrials"],
                    "restricted_sectors": []
                },
                "esg_preferences": {
                    "esg_integration": False,
                    "exclusion_screening": False,
                    "impact_investing": False,
                    "climate_focus": False,
                    "esg_minimum_rating": None,
                    "carbon_footprint_limit": False
                },
                "compliance_status": "active",
                "kyc_last_updated": "2025-01-20",
                "account_manager": "jane.smith@bank.com",
                "metadata": {
                    "onboarding_date": "2018-06-01",
                    "last_review_date": "2025-01-20",
                    "next_review_date": "2026-01-20",
                    "suitability_score": 9.1,
                    "complexity_level": "institutional"
                }
            },
            {
                "client_id": "CLIENT_003",
                "client_name": "Gamma Pension Fund",
                "client_type": "pension_fund",
                "incorporation_country": "Canada",
                "business_sector": "Pension/Insurance",
                "aum": 2000000000,
                "risk_tolerance": "conservative",
                "investment_objectives": {
                    "primary": "income",
                    "secondary": "capital_preservation",
                    "time_horizon": "long_term",
                    "return_target": 6.5,
                    "volatility_tolerance": "low"
                },
                "geographic_preferences": {
                    "home_bias": 70,
                    "developed_markets": 25,
                    "emerging_markets": 5,
                    "preferred_regions": ["North America", "Europe"],
                    "restricted_countries": ["High-risk jurisdictions"]
                },
                "asset_class_preferences": {
                    "equities": 35,
                    "fixed_income": 55,
                    "alternatives": 8,
                    "cash": 2,
                    "preferred_sectors": ["Utilities", "Consumer Staples", "Healthcare"],
                    "restricted_sectors": ["Speculative Growth", "Biotech"]
                },
                "esg_preferences": {
                    "esg_integration": True,
                    "exclusion_screening": True,
                    "impact_investing": True,
                    "climate_focus": True,
                    "esg_minimum_rating": "A",
                    "carbon_footprint_limit": True
                },
                "compliance_status": "active",
                "kyc_last_updated": "2025-02-10",
                "account_manager": "mike.wilson@bank.com",
                "metadata": {
                    "onboarding_date": "2015-03-01",
                    "last_review_date": "2025-02-10",
                    "next_review_date": "2026-02-10",
                    "suitability_score": 7.8,
                    "complexity_level": "institutional",
                    "regulatory_requirements": ["ERISA", "Canadian pension regulations"]
                }
            }
        ]
        
        # Filter by client_id if provided
        if client_id:
            mock_records = [record for record in mock_records if record.get("client_id") == client_id]
        
        self.logger.info(f"Generated {len(mock_records)} mock KYC records")
        return mock_records

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw KYC data.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed KYC records
        """
        if not raw_data:
            self.logger.warning("No raw KYC data provided for preprocessing")
            return []
        
        processed_records = []
        
        for record in raw_data:
            try:
                processed_record = self._process_single_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                self.logger.error(f"Error preprocessing KYC record {record.get('client_id')}: {str(e)}")
        
        self.logger.info(f"Preprocessed {len(processed_records)} KYC records")
        return processed_records

    def _process_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single KYC record."""
        processed_record = {
            "client_id": record.get("client_id"),
            "client_name": record.get("client_name"),
            "client_type": record.get("client_type", "").lower(),
            "incorporation_country": record.get("incorporation_country"),
            "business_sector": record.get("business_sector"),
            "aum": record.get("aum"),
            "risk_tolerance": record.get("risk_tolerance", "").lower(),
            "investment_objectives": record.get("investment_objectives", {}),
            "geographic_preferences": record.get("geographic_preferences", {}),
            "asset_class_preferences": record.get("asset_class_preferences", {}),
            "esg_preferences": record.get("esg_preferences", {}),
            "compliance_status": record.get("compliance_status", "").lower(),
            "kyc_last_updated": self._parse_date(record.get("kyc_last_updated")),
            "account_manager": record.get("account_manager"),
            "metadata": record.get("metadata", {})
        }
        
        # Validate required fields
        if not processed_record["client_id"]:
            return None
        
        if processed_record["compliance_status"] not in ["active", "inactive", "pending"]:
            processed_record["compliance_status"] = "unknown"
        
        return processed_record

    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats."""
        if not date_value:
            return None
        
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
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
        Build features from processed KYC data.
        
        Args:
            processed_data: Preprocessed KYC data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched KYC records
        """
        enriched_records = []
        
        for record in processed_data:
            try:
                enriched_record = record.copy()
                
                # Add derived features
                enriched_record.update(self._calculate_profile_features(record))
                
                # Extract investment themes using LLM if available
                if llm_client:
                    themes = self._extract_investment_themes(record, llm_client)
                    enriched_record["investment_themes"] = themes
                
                enriched_records.append(enriched_record)
                
            except Exception as e:
                self.logger.error(f"Error building features for KYC {record.get('client_id')}: {str(e)}")
                enriched_records.append(record)
        
        return enriched_records

    def _calculate_profile_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate profile-specific features."""
        features = {}
        
        # Risk scoring
        risk_tolerance = record.get("risk_tolerance", "")
        risk_score_map = {"conservative": 1, "moderate": 2, "aggressive": 3}
        features["risk_score"] = risk_score_map.get(risk_tolerance, 0)
        
        # AUM categorization
        aum = record.get("aum", 0)
        if aum >= 1000000000:  # $1B+
            features["aum_category"] = "ultra_high"
        elif aum >= 100000000:  # $100M+
            features["aum_category"] = "high"
        elif aum >= 10000000:   # $10M+
            features["aum_category"] = "medium"
        else:
            features["aum_category"] = "small"
        
        # ESG orientation
        esg_prefs = record.get("esg_preferences", {})
        esg_score = sum([
            esg_prefs.get("esg_integration", False),
            esg_prefs.get("exclusion_screening", False),
            esg_prefs.get("impact_investing", False),
            esg_prefs.get("climate_focus", False)
        ])
        features["esg_orientation_score"] = esg_score
        features["is_esg_focused"] = esg_score >= 2
        
        # Geographic diversification
        geo_prefs = record.get("geographic_preferences", {})
        home_bias = geo_prefs.get("home_bias", 100)
        features["home_bias_percentage"] = home_bias
        features["is_globally_diversified"] = home_bias < 60
        
        # Asset allocation analysis
        asset_prefs = record.get("asset_class_preferences", {})
        equity_allocation = asset_prefs.get("equities", 0)
        features["equity_allocation"] = equity_allocation
        features["is_equity_focused"] = equity_allocation > 60
        
        # Client sophistication
        client_type = record.get("client_type", "")
        sophistication_map = {
            "individual": 1, "corporate": 2, "institutional": 3, 
            "pension_fund": 3, "sovereign_wealth": 4
        }
        features["sophistication_score"] = sophistication_map.get(client_type, 1)
        
        # Profile freshness
        last_updated = record.get("kyc_last_updated")
        if last_updated:
            days_since_update = (datetime.now() - last_updated).days
            features["profile_freshness_days"] = days_since_update
            features["profile_needs_update"] = days_since_update > 365
        
        return features

    def _extract_investment_themes(self, record: Dict[str, Any], llm_client) -> List[str]:
        """Extract investment themes using LLM."""
        try:
            # Build context from KYC profile
            context = f"""
            Client: {record.get('client_name')} ({record.get('client_type')})
            Sector: {record.get('business_sector')}
            Risk Tolerance: {record.get('risk_tolerance')}
            Investment Objectives: {record.get('investment_objectives', {})}
            Asset Preferences: {record.get('asset_class_preferences', {})}
            ESG Preferences: {record.get('esg_preferences', {})}
            """
            
            prompt = f"""
            Based on this KYC profile, identify the top 5 investment themes that would be 
            most relevant for this client. Consider their risk tolerance, objectives, 
            sector preferences, and ESG orientation.
            
            {context}
            
            Return investment themes as comma-separated list:
            """
            
            response = llm_client.generate_completion(prompt, max_tokens=100)
            
            if hasattr(response, 'content'):
                response = response.content
            
            # Parse themes
            themes = [theme.strip() for theme in str(response).split(",")]
            return [theme for theme in themes if theme][:5]
            
        except Exception as e:
            self.logger.error(f"Error extracting investment themes: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """
        Test connection to KYC system.
        
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
                self.logger.info("KYC database connection test passed")
                return True
            
            if self.kyc_system_url and self.api_key:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(
                    f"{self.kyc_system_url}/api/v1/health",
                    headers=headers,
                    timeout=10
                )
                success = response.status_code == 200
                if success:
                    self.logger.info("KYC API connection test passed")
                else:
                    self.logger.error(f"KYC API connection test failed: {response.status_code}")
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"KYC connection test failed: {str(e)}")
            return False

    def get_client_investment_profile(self, client_id: str) -> Dict[str, Any]:
        """
        Get comprehensive investment profile for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Comprehensive investment profile
        """
        try:
            records = self.ingest(client_id=client_id)
            
            if not records:
                return {"client_id": client_id, "profile_found": False}
            
            profile = records[0]  # Should be only one record per client
            processed_profile = self.preprocess([profile])[0]
            enriched_profile = self.build_features([processed_profile])[0]
            
            return {
                "client_id": client_id,
                "profile_found": True,
                **enriched_profile
            }
            
        except Exception as e:
            self.logger.error(f"Error getting investment profile for {client_id}: {str(e)}")
            return {"client_id": client_id, "error": str(e)} 