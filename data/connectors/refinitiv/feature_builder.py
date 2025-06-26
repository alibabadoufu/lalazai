"""
Refinitiv Chat Feature Builder.
This module uses LLM to extract features from Refinitiv chat data with financial market focus.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class RefinitivChatFeatureBuilder:
    """
    Feature builder for Refinitiv chat data.
    
    Uses LLM to extract meaningful features from Refinitiv chat messages including:
    - Market-focused conversation summaries
    - Financial keywords and instruments
    - Market sentiment analysis
    - Trading intent signals
    - Currency pair and instrument mentions
    """

    def __init__(self):
        """Initialize the feature builder."""
        self.logger = logging.getLogger(__name__)
        
        # Prompts tailored for Refinitiv financial chat
        self.prompts = {
            "summary": self._get_summary_prompt(),
            "keywords": self._get_keywords_prompt(),
            "sentiment": self._get_sentiment_prompt(),
            "intent": self._get_intent_prompt(),
            "instruments": self._get_instruments_prompt()
        }

    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Build features from processed Refinitiv chat data.
        
        Args:
            processed_data: Preprocessed chat data
            llm_client: LLM client for feature extraction
            
        Returns:
            List of feature-enriched chat records
        """
        if not processed_data:
            self.logger.warning("No processed data provided for feature building")
            return []

        enriched_records = []
        
        for record in processed_data:
            try:
                enriched_record = self._build_single_record_features(record, llm_client)
                enriched_records.append(enriched_record)
            except Exception as e:
                self.logger.error(f"Error building features for record {record.get('message_id')}: {str(e)}")
                # Add record without LLM features
                enriched_records.append(self._add_fallback_features(record))

        self.logger.info(f"Built features for {len(enriched_records)} Refinitiv chat records")
        return enriched_records

    def _build_single_record_features(self, record: Dict[str, Any], 
                                    llm_client=None) -> Dict[str, Any]:
        """
        Build features for a single Refinitiv chat record.
        
        Args:
            record: Single processed chat record
            llm_client: LLM client for feature extraction
            
        Returns:
            Feature-enriched record
        """
        # Start with the original record
        enriched_record = record.copy()
        
        message_content = record.get("message_content", "")
        
        if not message_content.strip():
            return self._add_fallback_features(enriched_record)

        if llm_client:
            # Extract features using LLM
            enriched_record.update(self._extract_llm_features(message_content, llm_client))
        else:
            # Use rule-based fallback features
            enriched_record.update(self._extract_rule_based_features(message_content))

        # Add metadata
        enriched_record["feature_extraction_timestamp"] = datetime.now().isoformat()
        enriched_record["feature_extraction_method"] = "llm" if llm_client else "rule_based"

        return enriched_record

    def _extract_llm_features(self, message_content: str, llm_client) -> Dict[str, Any]:
        """
        Extract features using LLM with Refinitiv-specific prompts.
        
        Args:
            message_content: Message content to analyze
            llm_client: LLM client
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Extract market-focused summary
            summary_prompt = self.prompts["summary"].format(message=message_content)
            summary = llm_client.generate_completion(summary_prompt, max_tokens=100)
            features["summary"] = self._clean_llm_output(summary)
            
            # Extract financial keywords
            keywords_prompt = self.prompts["keywords"].format(message=message_content)
            keywords_response = llm_client.generate_completion(keywords_prompt, max_tokens=50)
            features["keywords"] = self._parse_keywords(keywords_response)
            
            # Extract market sentiment
            sentiment_prompt = self.prompts["sentiment"].format(message=message_content)
            sentiment_response = llm_client.generate_completion(sentiment_prompt, max_tokens=20)
            features["sentiment"] = self._parse_sentiment(sentiment_response)
            
            # Extract trading intent signals
            intent_prompt = self.prompts["intent"].format(message=message_content)
            intent_response = llm_client.generate_completion(intent_prompt, max_tokens=100)
            features["intent_signals"] = self._parse_intent_signals(intent_response)
            
            # Extract financial instruments
            instruments_prompt = self.prompts["instruments"].format(message=message_content)
            instruments_response = llm_client.generate_completion(instruments_prompt, max_tokens=50)
            features["instrument_mentions"] = self._parse_instrument_mentions(instruments_response)
            
        except Exception as e:
            self.logger.error(f"Error extracting LLM features: {str(e)}")
            # Fall back to rule-based features
            features.update(self._extract_rule_based_features(message_content))
        
        return features

    def _extract_rule_based_features(self, message_content: str) -> Dict[str, Any]:
        """
        Extract features using rule-based methods as fallback.
        
        Args:
            message_content: Message content to analyze
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            "summary": self._create_rule_based_summary(message_content),
            "keywords": self._extract_rule_based_keywords(message_content),
            "sentiment": self._detect_rule_based_sentiment(message_content),
            "intent_signals": self._detect_rule_based_intent(message_content),
            "instrument_mentions": self._detect_rule_based_instruments(message_content)
        }
        
        return features

    def _extract_rule_based_keywords(self, message_content: str) -> List[str]:
        """Extract financial keywords using rules."""
        # Financial market keywords for Refinitiv context
        financial_keywords = [
            # FX
            "eur/usd", "gbp/usd", "usd/jpy", "aud/usd", "usd/cad", "eur/gbp", "dovish", "hawkish",
            # Commodities
            "oil", "gold", "silver", "copper", "wti", "brent", "inventories",
            # Market terms
            "bullish", "bearish", "volatility", "support", "resistance", "breakout", "target",
            "rally", "sell-off", "correction", "bounce", "momentum", "volume",
            # Economic data
            "gdp", "inflation", "employment", "unemployment", "cpi", "ppi", "fed", "ecb", "boe",
            # Trading
            "long", "short", "position", "hedge", "exposure", "risk", "profit", "loss"
        ]
        
        content_lower = message_content.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:10]  # Limit to 10 keywords

    def _detect_rule_based_sentiment(self, message_content: str) -> str:
        """Detect market sentiment using rules."""
        content_lower = message_content.lower()
        
        # Bullish indicators
        bullish_words = ["bullish", "rally", "up", "rising", "strong", "buy", "positive", "gains", "breakout", "support"]
        # Bearish indicators
        bearish_words = ["bearish", "sell-off", "down", "falling", "weak", "sell", "negative", "losses", "breakdown", "resistance"]
        
        bullish_score = sum(1 for word in bullish_words if word in content_lower)
        bearish_score = sum(1 for word in bearish_words if word in content_lower)
        
        if bullish_score > bearish_score:
            return "bullish"
        elif bearish_score > bullish_score:
            return "bearish"
        else:
            return "neutral"

    def _detect_rule_based_intent(self, message_content: str) -> List[str]:
        """Detect trading intent signals using rules."""
        content_lower = message_content.lower()
        intents = []
        
        # Trading intent patterns
        if any(word in content_lower for word in ["buy", "long", "bullish", "target"]):
            intents.append("buy_intent")
        
        if any(word in content_lower for word in ["sell", "short", "bearish", "stop"]):
            intents.append("sell_intent")
        
        if any(word in content_lower for word in ["watch", "monitor", "view", "think", "expect"]):
            intents.append("market_view")
        
        if any(word in content_lower for word in ["data", "news", "report", "announcement"]):
            intents.append("information_sharing")
        
        if "?" in message_content:
            intents.append("question")
        
        return intents

    def _detect_rule_based_instruments(self, message_content: str) -> List[str]:
        """Detect financial instruments using rules."""
        content_upper = message_content.upper()
        instruments = []
        
        # Currency pairs
        currency_patterns = [
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "EUR/GBP",
            "EUR=", "GBP=", "JPY=", "AUD=", "CAD="
        ]
        for pattern in currency_patterns:
            if pattern in content_upper:
                # Normalize to standard format
                if "=" in pattern:
                    instruments.append(pattern.replace("=", "/USD"))
                else:
                    instruments.append(pattern)
        
        # Commodities
        commodity_patterns = ["WTI", "BRENT", "GOLD", "SILVER", "COPPER", "OIL"]
        for commodity in commodity_patterns:
            if commodity in content_upper:
                instruments.append(commodity)
        
        # Indices (common Refinitiv codes)
        index_patterns = [".SPX", ".FTSE", ".N225", ".HSI", "SPX", "FTSE", "NIKKEI"]
        for index in index_patterns:
            if index in content_upper:
                instruments.append(index)
        
        return list(set(instruments))  # Remove duplicates

    def _add_fallback_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add minimal fallback features when processing fails."""
        record.update({
            "summary": "Feature extraction failed",
            "keywords": [],
            "sentiment": "unknown",
            "intent_signals": [],
            "instrument_mentions": [],
            "feature_extraction_timestamp": datetime.now().isoformat(),
            "feature_extraction_method": "fallback"
        })
        return record

    # LLM prompt templates for Refinitiv financial chat
    def _get_summary_prompt(self) -> str:
        return """
        Summarize this Refinitiv financial chat message in 1-2 sentences, focusing on market views, trading ideas, and financial instruments discussed:

        Message: {message}

        Summary:
        """

    def _get_keywords_prompt(self) -> str:
        return """
        Extract the top 5 most important financial keywords from this Refinitiv chat message. Focus on currency pairs, commodities, market terms, and trading concepts.

        Message: {message}

        Keywords (comma-separated):
        """

    def _get_sentiment_prompt(self) -> str:
        return """
        Analyze the market sentiment of this Refinitiv chat message. Respond with one word: bullish, bearish, or neutral.

        Message: {message}

        Market Sentiment:
        """

    def _get_intent_prompt(self) -> str:
        return """
        Identify the trading intent or purpose of this Refinitiv chat message. Common intents include: buy_intent, sell_intent, market_view, information_sharing, question, price_inquiry.

        Message: {message}

        Trading intent signals (comma-separated):
        """

    def _get_instruments_prompt(self) -> str:
        return """
        Identify any financial instruments mentioned in this Refinitiv chat message, including currency pairs, commodities, indices, or bonds.

        Message: {message}

        Financial instruments mentioned (comma-separated):
        """

    # Helper methods for parsing LLM outputs
    def _clean_llm_output(self, output: str) -> str:
        """Clean and normalize LLM output."""
        if hasattr(output, 'content'):
            output = output.content
        
        if not output:
            return ""
        
        # Remove common LLM artifacts
        output = str(output).strip()
        output = output.replace("Summary:", "").replace("summary:", "")
        output = output.replace("Market Sentiment:", "").replace("sentiment:", "")
        
        return output.strip()

    def _parse_keywords(self, output: str) -> List[str]:
        """Parse keywords from LLM output."""
        if hasattr(output, 'content'):
            output = output.content
            
        if not output:
            return []
        
        # Clean and split by commas
        keywords = [kw.strip().lower() for kw in str(output).split(",")]
        # Remove empty strings and limit to 10
        keywords = [kw for kw in keywords if kw][:10]
        
        return keywords

    def _parse_sentiment(self, output: str) -> str:
        """Parse market sentiment from LLM output."""
        if hasattr(output, 'content'):
            output = output.content
            
        if not output:
            return "unknown"
        
        output_lower = str(output).lower().strip()
        
        if "bullish" in output_lower:
            return "bullish"
        elif "bearish" in output_lower:
            return "bearish"
        elif "neutral" in output_lower:
            return "neutral"
        else:
            return "unknown"

    def _parse_intent_signals(self, output: str) -> List[str]:
        """Parse trading intent signals from LLM output."""
        if hasattr(output, 'content'):
            output = output.content
            
        if not output:
            return []
        
        # Split by commas and clean
        intents = [intent.strip().lower() for intent in str(output).split(",")]
        intents = [intent for intent in intents if intent][:5]  # Limit to 5
        
        return intents

    def _parse_instrument_mentions(self, output: str) -> List[str]:
        """Parse financial instrument mentions from LLM output."""
        if hasattr(output, 'content'):
            output = output.content
            
        if not output:
            return []
        
        # Split by commas and clean
        instruments = [instrument.strip().upper() for instrument in str(output).split(",")]
        instruments = [instrument for instrument in instruments if instrument][:10]  # Limit to 10
        
        return instruments

    def _create_rule_based_summary(self, message_content: str) -> str:
        """Create a simple rule-based summary."""
        # Take first 100 characters as summary
        summary = message_content[:100]
        if len(message_content) > 100:
            summary += "..."
        return f"Refinitiv chat: {summary}" 