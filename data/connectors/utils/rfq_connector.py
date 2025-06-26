"""
RFQ (Request for Quote) Connector Implementation.
This module handles RFQ data ingestion for trading requests and client transaction preferences.
"""

import os
import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

from .base_connector import BaseConnector
from .schemas import RFQData, ConnectorConfig


class RFQConnector(BaseConnector):
    """
    Connector for RFQ data including trading requests, instrument preferences, and transaction history.
    
    Captures client trading intent and preferences to inform research recommendations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RFQ connector.
        
        Args:
            config: Configuration dictionary containing RFQ settings
        """
        # Set default connector name if not provided
        if "connector_name" not in config:
            config["connector_name"] = "rfq"
            
        super().__init__(config)
        
        # RFQ system configuration
        self.trading_system_url = config.get("trading_system_url", os.getenv("TRADING_SYSTEM_URL"))
        self.api_key = config.get("api_key", os.getenv("TRADING_API_KEY"))
        self.database_path = config.get("database_path", os.getenv("RFQ_DATABASE_PATH"))
        self.timeout = config.get("timeout", 30)
        
        # Database connection for local RFQ data
        self.db_connection = None
        
        if not self.trading_system_url and not self.database_path:
            self.logger.warning("No RFQ data source configured. Using mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False
            
        if self.database_path:
            self._init_database()

    def _init_database(self):
        """Initialize database connection for local RFQ data."""
        try:
            self.db_connection = sqlite3.connect(self.database_path)
            self.db_connection.row_factory = sqlite3.Row
            self.logger.info(f"Connected to RFQ database: {self.database_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to RFQ database: {str(e)}")
            self.use_mock_data = True

    def ingest(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Ingest RFQ data.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional parameters (instrument_type, min_notional, etc.)
            
        Returns:
            List of raw RFQ records
            
        Raises:
            ConnectionError: If unable to connect to trading system
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
            self.logger.error(f"Error ingesting RFQ data: {str(e)}")
            raise ConnectionError(f"Failed to ingest RFQ data: {str(e)}") from e

    def _fetch_from_api(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from trading system API.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional API parameters
            
        Returns:
            List of raw RFQ records
        """
        lookback_date = self.get_lookback_date()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "start_date": lookback_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "include_rejected": True,
            "include_expired": True
        }
        
        if client_id:
            params["client_id"] = client_id
            
        # Add additional filters
        for key, value in kwargs.items():
            if key in ["instrument_type", "min_notional", "currency", "status"]:
                params[key] = value
        
        try:
            response = requests.get(
                f"{self.trading_system_url}/api/v1/rfqs",
                params=params,  
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            records = data.get("rfqs", [])
            
            self.logger.info(f"Fetched {len(records)} RFQ records from trading system API")
            return records
            
        except requests.RequestException as e:
            raise ConnectionError(f"Trading system API request failed: {str(e)}") from e

    def _fetch_from_database(self, client_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from local RFQ database.
        
        Args:
            client_id: Optional client ID to filter records
            **kwargs: Additional database parameters
            
        Returns:
            List of raw RFQ records
        """
        if not self.db_connection:
            raise ConnectionError("Database connection not available")
        
        lookback_date = self.get_lookback_date()
        
        query = """
        SELECT 
            rfq_id, client_id, instrument_type, instrument_symbol,
            side, notional_amount, currency, requested_price,
            actual_price, status, request_timestamp, 
            expiry_timestamp, execution_timestamp,
            trader_id, desk, reason_code, metadata
        FROM rfq_requests 
        WHERE request_timestamp >= ?
        """
        
        params = [lookback_date.strftime('%Y-%m-%d %H:%M:%S')]
        
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        
        # Add additional filters
        if kwargs.get("instrument_type"):
            query += " AND instrument_type = ?"
            params.append(kwargs["instrument_type"])
            
        if kwargs.get("min_notional"):
            query += " AND notional_amount >= ?"
            params.append(kwargs["min_notional"])
        
        query += " ORDER BY request_timestamp DESC"
        
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
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
            
            self.logger.info(f"Fetched {len(records)} RFQ records from database")
            return records
            
        except sqlite3.Error as e:
            raise ConnectionError(f"Database query failed: {str(e)}") from e

    def _get_mock_data(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate mock RFQ data for testing.
        
        Args:
            client_id: Optional client ID to filter records
            
        Returns:
            List of mock RFQ records
        """
        mock_records = [
            {
                "rfq_id": "RFQ_001",
                "client_id": "CLIENT_001",
                "client_name": "Acme Corp",
                "instrument_type": "equity",
                "instrument_symbol": "AAPL",
                "side": "buy",
                "notional_amount": 10000000,
                "currency": "USD",
                "requested_price": None,
                "actual_price": 150.25,
                "status": "executed",
                "request_timestamp": "2025-06-21 10:30:00",
                "expiry_timestamp": "2025-06-21 16:00:00",
                "execution_timestamp": "2025-06-21 10:32:15",
                "trader_id": "trader_001",
                "desk": "US_EQUITY",
                "reason_code": "client_rebalancing",
                "metadata": {
                    "execution_venue": "NYSE",
                    "algo_strategy": "TWAP",
                    "benchmark": "VWAP",
                    "urgency": "normal"
                }
            },
            {
                "rfq_id": "RFQ_002",
                "client_id": "CLIENT_002",
                "client_name": "Beta Investment Partners",
                "instrument_type": "fx",
                "instrument_symbol": "EUR/USD",
                "side": "sell",
                "notional_amount": 25000000,
                "currency": "EUR",
                "requested_price": 1.0850,
                "actual_price": 1.0848,
                "status": "executed",
                "request_timestamp": "2025-06-21 09:15:00",
                "expiry_timestamp": "2025-06-21 15:00:00",
                "execution_timestamp": "2025-06-21 09:16:30",
                "trader_id": "trader_002",
                "desk": "FX_SPOT",
                "reason_code": "hedging",
                "metadata": {
                    "execution_venue": "EBS",
                    "hedge_ratio": 0.75,
                    "underlying_position": "EUR_BONDS"
                }
            },
            {
                "rfq_id": "RFQ_003",
                "client_id": "CLIENT_001", 
                "client_name": "Acme Corp",
                "instrument_type": "fixed_income",
                "instrument_symbol": "US10Y",
                "side": "buy",
                "notional_amount": 50000000,
                "currency": "USD",
                "requested_price": None,
                "actual_price": None,
                "status": "expired",
                "request_timestamp": "2025-06-20 14:30:00",
                "expiry_timestamp": "2025-06-20 16:00:00",
                "execution_timestamp": None,
                "trader_id": "trader_003",
                "desk": "RATES",
                "reason_code": "duration_extension",
                "metadata": {
                    "requested_duration": "10Y",
                    "yield_target": "4.25%",
                    "expiry_reason": "no_competitive_quotes"
                }
            },
            {
                "rfq_id": "RFQ_004",
                "client_id": "CLIENT_003",
                "client_name": "Gamma Pension Fund",
                "instrument_type": "commodity",
                "instrument_symbol": "GOLD",
                "side": "buy",
                "notional_amount": 5000000,
                "currency": "USD",
                "requested_price": 1950,
                "actual_price": None,
                "status": "rejected",
                "request_timestamp": "2025-06-19 11:45:00",
                "expiry_timestamp": "2025-06-19 17:00:00",
                "execution_timestamp": None,
                "trader_id": "trader_004",
                "desk": "COMMODITIES",
                "reason_code": "portfolio_diversification",
                "metadata": {
                    "rejection_reason": "price_out_of_range",
                    "market_price": 1965,
                    "requested_discount": "0.75%"
                }
            },
            {
                "rfq_id": "RFQ_005",
                "client_id": "CLIENT_002",
                "client_name": "Beta Investment Partners",
                "instrument_type": "equity",
                "instrument_symbol": "TSLA",
                "side": "sell",
                "notional_amount": 8000000,
                "currency": "USD", 
                "requested_price": 220,
                "actual_price": 218.50,
                "status": "executed",
                "request_timestamp": "2025-06-18 13:20:00",
                "expiry_timestamp": "2025-06-18 16:00:00",
                "execution_timestamp": "2025-06-18 13:22:45",
                "trader_id": "trader_001",
                "desk": "US_EQUITY",
                "reason_code": "profit_taking",
                "metadata": {
                    "execution_venue": "NASDAQ",
                    "block_size": "large",
                    "market_impact": "minimal"
                }
            }
        ]
        
        # Filter by client_id if provided
        if client_id:
            mock_records = [record for record in mock_records if record.get("client_id") == client_id]
        
        self.logger.info(f"Generated {len(mock_records)} mock RFQ records")
        return mock_records

    def preprocess(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess raw RFQ data.
        
        Args:
            raw_data: Raw data from ingestion step
            
        Returns:
            List of preprocessed RFQ records
        """
        if not raw_data:
            self.logger.warning("No raw RFQ data provided for preprocessing")
            return []
        
        processed_records = []
        
        for record in raw_data:
            try:
                processed_record = self._process_single_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                self.logger.error(f"Error preprocessing RFQ record {record.get('rfq_id')}: {str(e)}")
        
        self.logger.info(f"Preprocessed {len(processed_records)} RFQ records")
        return processed_records

    def _process_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single RFQ record."""
        processed_record = {
            "rfq_id": record.get("rfq_id"),
            "client_id": record.get("client_id"),
            "client_name": record.get("client_name"),
            "instrument_type": record.get("instrument_type", "").lower(),
            "instrument_symbol": record.get("instrument_symbol", "").upper(),
            "side": record.get("side", "").lower(),
            "notional_amount": self._parse_decimal(record.get("notional_amount")),
            "currency": record.get("currency", "").upper(),
            "requested_price": self._parse_decimal(record.get("requested_price")),
            "actual_price": self._parse_decimal(record.get("actual_price")),
            "status": record.get("status", "").lower(),
            "request_timestamp": self._parse_timestamp(record.get("request_timestamp")),
            "expiry_timestamp": self._parse_timestamp(record.get("expiry_timestamp")),
            "execution_timestamp": self._parse_timestamp(record.get("execution_timestamp")),
            "trader_id": record.get("trader_id"),
            "desk": record.get("desk"),
            "reason_code": record.get("reason_code"),
            "metadata": record.get("metadata", {})
        }
        
        # Validate required fields
        if not processed_record["rfq_id"] or not processed_record["client_id"]:
            return None
        
        if not processed_record["request_timestamp"]:
            return None
        
        return processed_record

    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        """Parse decimal values safely."""
        if value is None:
            return None
        
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            return None

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
        Build features from processed RFQ data.
        
        Args:
            processed_data: Preprocessed RFQ data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched RFQ records
        """
        enriched_records = []
        
        for record in processed_data:
            try:
                enriched_record = record.copy()
                
                # Add derived trading features
                enriched_record.update(self._calculate_trading_features(record))
                
                # Extract trading intent using LLM if available
                if llm_client and record.get("reason_code"):
                    intent = self._extract_trading_intent(record, llm_client)
                    enriched_record["trading_intent"] = intent
                
                enriched_records.append(enriched_record)
                
            except Exception as e:
                self.logger.error(f"Error building features for RFQ {record.get('rfq_id')}: {str(e)}")
                enriched_records.append(record)
        
        return enriched_records

    def _calculate_trading_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trading-specific features."""
        features = {}
        
        # Time-based features
        request_time = record.get("request_timestamp")
        expiry_time = record.get("expiry_timestamp")
        execution_time = record.get("execution_timestamp")
        
        if request_time and expiry_time:
            features["time_to_expiry_minutes"] = int((expiry_time - request_time).total_seconds() / 60)
        
        if request_time and execution_time:
            features["execution_time_seconds"] = int((execution_time - request_time).total_seconds())
        
        # Price features
        requested_price = record.get("requested_price")
        actual_price = record.get("actual_price")
        
        if requested_price and actual_price:
            price_diff = actual_price - requested_price
            features["price_improvement"] = float(price_diff)
            features["price_improvement_pct"] = float(price_diff / requested_price * 100)
        
        # Size categorization
        notional = record.get("notional_amount")
        if notional:
            if notional >= 100000000:  # $100M+
                features["size_category"] = "block"
            elif notional >= 10000000:  # $10M+
                features["size_category"] = "large"
            elif notional >= 1000000:   # $1M+
                features["size_category"] = "medium"
            else:
                features["size_category"] = "small"
        
        # Trading pattern features
        features["is_buy"] = record.get("side") == "buy"
        features["is_executed"] = record.get("status") == "executed"
        features["is_risk_reducing"] = record.get("reason_code") in ["hedging", "risk_reduction"]
        
        return features

    def _extract_trading_intent(self, record: Dict[str, Any], llm_client) -> str:
        """Extract trading intent using LLM."""
        try:
            prompt = f"""
            Analyze this RFQ record and determine the primary trading intent. 
            Consider the instrument, side, reason code, and context.
            
            Instrument: {record.get('instrument_symbol')} ({record.get('instrument_type')})
            Side: {record.get('side')}
            Notional: {record.get('notional_amount')} {record.get('currency')}
            Reason: {record.get('reason_code')}
            
            Classify the intent as one of: speculation, hedging, rebalancing, 
            profit_taking, position_building, risk_reduction, liquidity_management
            
            Trading Intent:
            """
            
            response = llm_client.generate_completion(prompt, max_tokens=50)
            
            if hasattr(response, 'content'):
                response = response.content
            
            intent = str(response).strip().lower()
            
            # Validate intent
            valid_intents = [
                "speculation", "hedging", "rebalancing", "profit_taking",
                "position_building", "risk_reduction", "liquidity_management"
            ]
            
            for valid_intent in valid_intents:
                if valid_intent in intent:
                    return valid_intent
            
            return "unknown"
            
        except Exception as e:
            self.logger.error(f"Error extracting trading intent: {str(e)}")
            return "unknown"

    def test_connection(self) -> bool:
        """
        Test connection to RFQ system.
        
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
                self.logger.info("RFQ database connection test passed")
                return True
            
            if self.trading_system_url and self.api_key:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(
                    f"{self.trading_system_url}/api/v1/health",
                    headers=headers,
                    timeout=10
                )
                success = response.status_code == 200
                if success:
                    self.logger.info("RFQ API connection test passed")
                else:
                    self.logger.error(f"RFQ API connection test failed: {response.status_code}")
                return success
            
            return False
            
        except Exception as e:
            self.logger.error(f"RFQ connection test failed: {str(e)}")
            return False

    def get_client_trading_profile(self, client_id: str) -> Dict[str, Any]:
        """
        Get trading profile for a specific client based on RFQ history.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client trading profile
        """
        try:
            records = self.ingest(client_id=client_id)
            processed_records = self.preprocess(records)
            
            if not processed_records:
                return {"client_id": client_id, "total_rfqs": 0}
            
            # Calculate trading statistics
            total_rfqs = len(processed_records)
            executed_rfqs = [r for r in processed_records if r.get("status") == "executed"]
            execution_rate = len(executed_rfqs) / total_rfqs if total_rfqs > 0 else 0
            
            # Instrument preferences
            instrument_counts = {}
            for record in processed_records:
                instrument_type = record.get("instrument_type", "unknown")
                instrument_counts[instrument_type] = instrument_counts.get(instrument_type, 0) + 1
            
            # Side bias
            buy_count = sum(1 for r in processed_records if r.get("side") == "buy")
            sell_count = sum(1 for r in processed_records if r.get("side") == "sell")
            
            # Average notional
            notionals = [r.get("notional_amount") for r in processed_records 
                        if r.get("notional_amount")]
            avg_notional = sum(notionals) / len(notionals) if notionals else 0
            
            profile = {
                "client_id": client_id,
                "total_rfqs": total_rfqs,
                "execution_rate": round(execution_rate, 3),
                "preferred_instruments": dict(sorted(instrument_counts.items(), 
                                                   key=lambda x: x[1], reverse=True)),
                "side_bias": {
                    "buy_percentage": round(buy_count / total_rfqs * 100, 1) if total_rfqs > 0 else 0,
                    "sell_percentage": round(sell_count / total_rfqs * 100, 1) if total_rfqs > 0 else 0
                },
                "average_notional": float(avg_notional) if avg_notional else 0,
                "last_rfq_date": max([r.get("request_timestamp") for r in processed_records 
                                    if r.get("request_timestamp")]) if processed_records else None
            }
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting trading profile for {client_id}: {str(e)}")
            return {"client_id": client_id, "error": str(e)} 