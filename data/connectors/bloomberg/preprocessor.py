"""
Bloomberg Chat Data Preprocessor.
This module handles cleaning and standardization of raw Bloomberg chat data.
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import ValidationError

from ..schemas import ChatMessage


class BloombergChatPreprocessor:
    """
    Preprocessor for Bloomberg chat data.
    
    Handles data cleaning, standardization, and validation.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self.logger = logging.getLogger(__name__)

    def process(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw Bloomberg chat data.
        
        Args:
            raw_data: Raw chat data from ingestion
            
        Returns:
            List of cleaned and standardized chat records
            
        Raises:
            ValueError: If preprocessing fails
        """
        if not raw_data:
            self.logger.warning("No raw data provided for preprocessing")
            return []

        processed_records = []
        errors = []

        for i, record in enumerate(raw_data):
            try:
                processed_record = self._process_single_record(record)
                if processed_record:
                    processed_records.append(processed_record)
            except Exception as e:
                error_msg = f"Error processing record {i}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        if errors:
            self.logger.warning(f"Preprocessing completed with {len(errors)} errors out of {len(raw_data)} records")

        self.logger.info(f"Successfully preprocessed {len(processed_records)} records")
        return processed_records

    def _process_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single chat record.
        
        Args:
            record: Single raw chat record
            
        Returns:
            Processed record or None if invalid
        """
        # Clean and standardize the record
        cleaned_record = {
            "message_id": self._clean_string(record.get("message_id", "")),
            "timestamp": self._parse_timestamp(record.get("timestamp")),
            "participants": self._parse_participants(record.get("participants", [])),
            "message_content": self._clean_message_content(record.get("message_content", "")),
            "client_id": self._clean_string(record.get("client_id")),
            "conversation_id": self._clean_string(record.get("conversation_id")),
            "message_type": record.get("message_type", "text"),
            "channel": record.get("channel", "bloomberg_chat"),
            "metadata": record.get("metadata", {})
        }

        # Validate required fields
        if not cleaned_record["message_content"].strip():
            self.logger.warning(f"Skipping record with empty message content: {record.get('message_id')}")
            return None

        if not cleaned_record["timestamp"]:
            self.logger.warning(f"Skipping record with invalid timestamp: {record.get('message_id')}")
            return None

        # Additional validation
        if not self._is_valid_record(cleaned_record):
            return None

        return cleaned_record

    def _clean_string(self, value: Any) -> Optional[str]:
        """
        Clean and normalize string values.
        
        Args:
            value: String value to clean
            
        Returns:
            Cleaned string or None
        """
        if value is None:
            return None
            
        if not isinstance(value, str):
            value = str(value)
            
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', value.strip())
        
        return cleaned if cleaned else None

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """
        Parse timestamp from various formats.
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            Parsed datetime or None
        """
        if timestamp is None:
            return None

        if isinstance(timestamp, datetime):
            return timestamp

        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError):
                return None

        if isinstance(timestamp, str):
            # Try different formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%d/%m/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            
            self.logger.warning(f"Unable to parse timestamp: {timestamp}")
            return None

        return None

    def _parse_participants(self, participants: Any) -> List[str]:
        """
        Parse and clean participant information.
        
        Args:
            participants: Participants data in various formats
            
        Returns:
            List of participant names
        """
        if not participants:
            return []

        if isinstance(participants, str):
            # Split by common delimiters
            participants = re.split(r'[,;|]', participants)

        if not isinstance(participants, list):
            return []

        # Clean each participant name
        cleaned_participants = []
        for participant in participants:
            cleaned = self._clean_string(participant)
            if cleaned:
                cleaned_participants.append(cleaned)

        return cleaned_participants

    def _clean_message_content(self, content: Any) -> str:
        """
        Clean and normalize message content.
        
        Args:
            content: Raw message content
            
        Returns:
            Cleaned message content
        """
        if content is None:
            return ""

        if not isinstance(content, str):
            content = str(content)

        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove or replace special characters that might cause issues
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)  # Remove control characters
        
        # Normalize quotes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace(''', "'").replace(''', "'")
        
        return content

    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a processed record.
        
        Args:
            record: Processed record to validate
            
        Returns:
            True if record is valid, False otherwise
        """
        # Check message length
        message_content = record.get("message_content", "")
        if len(message_content) > 10000:  # Arbitrary max length
            self.logger.warning(f"Message content too long ({len(message_content)} chars), truncating")
            record["message_content"] = message_content[:10000] + "..."

        # Check if timestamp is too old or in the future
        timestamp = record.get("timestamp")
        if timestamp and isinstance(timestamp, datetime):
            now = datetime.now()
            if timestamp > now:
                self.logger.warning(f"Future timestamp detected: {timestamp}")
                # Don't reject, but log the issue
            
            # Check if too old (more than 5 years)
            if (now - timestamp).days > 5 * 365:
                self.logger.warning(f"Very old timestamp detected: {timestamp}")

        # Validate participants
        participants = record.get("participants", [])
        if not participants:
            self.logger.warning(f"No participants found in message: {record.get('message_id')}")
            # Don't reject, but log the issue

        return True 