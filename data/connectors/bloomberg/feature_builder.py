"""
Bloomberg Chat Feature Builder.
This module uses LLM to extract summaries, topics, sentiment, and other features from chat data.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class BloombergChatFeatureBuilder:
    """
    Feature builder for Bloomberg chat data.
    
    Uses LLM to extract meaningful features from chat messages including:
    - Conversation summaries
    - Keywords and topics
    - Sentiment analysis
    - Intent signals
    - Product/currency mentions
    """

    def __init__(self):
        """Initialize the feature builder."""
        self.logger = logging.getLogger(__name__)
        
        # Prompts for different feature extraction tasks
        self.prompts = {
            "summary": self._get_summary_prompt(),
            "keywords": self._get_keywords_prompt(),
            "sentiment": self._get_sentiment_prompt(),
            "intent": self._get_intent_prompt(),
            "products": self._get_products_prompt()
        }

    def build_features(self, processed_data: List[Dict[str, Any]], 
                      llm_client=None) -> List[Dict[str, Any]]:
        """
        Build features from processed Bloomberg chat data.
        
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

        self.logger.info(f"Built features for {len(enriched_records)} chat records")
        return enriched_records

    def _build_single_record_features(self, record: Dict[str, Any], 
                                    llm_client=None) -> Dict[str, Any]:
        """
        Build features for a single chat record.
        
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
        Extract features using LLM.
        
        Args:
            message_content: Message content to analyze
            llm_client: LLM client
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        try:
            # Extract summary
            summary_prompt = self.prompts["summary"].format(message=message_content)
            summary = llm_client.generate_completion(summary_prompt, max_tokens=100)
            features["summary"] = self._clean_llm_output(summary)
            
            # Extract keywords
            keywords_prompt = self.prompts["keywords"].format(message=message_content)
            keywords_response = llm_client.generate_completion(keywords_prompt, max_tokens=50)
            features["keywords"] = self._parse_keywords(keywords_response)
            
            # Extract sentiment
            sentiment_prompt = self.prompts["sentiment"].format(message=message_content)
            sentiment_response = llm_client.generate_completion(sentiment_prompt, max_tokens=20)
            features["sentiment"] = self._parse_sentiment(sentiment_response)
            
            # Extract intent signals
            intent_prompt = self.prompts["intent"].format(message=message_content)
            intent_response = llm_client.generate_completion(intent_prompt, max_tokens=100)
            features["intent_signals"] = self._parse_intent_signals(intent_response)
            
            # Extract product/currency mentions
            products_prompt = self.prompts["products"].format(message=message_content)
            products_response = llm_client.generate_completion(products_prompt, max_tokens=50)
            features["product_mentions"] = self._parse_product_mentions(products_response)
            
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
            "product_mentions": self._detect_rule_based_products(message_content)
        }
        
        return features

    def _create_rule_based_summary(self, message_content: str) -> str:
        """Create a simple rule-based summary."""
        # Take first 100 characters as summary
        summary = message_content[:100]
        if len(message_content) > 100:
            summary += "..."
        return summary

    def _extract_rule_based_keywords(self, message_content: str) -> List[str]:
        """Extract keywords using simple rules."""
        # Common financial terms and keywords
        financial_keywords = [
            "costs", "tech", "solutions", "automation", "logistics", "supply chain",
            "rates", "volatile", "analysis", "trends", "research", "sector",
            "investment", "trading", "market", "price", "currency", "forex",
            "equity", "bond", "derivative", "option", "swap", "forward"
        ]
        
        content_lower = message_content.lower()
        found_keywords = []
        
        for keyword in financial_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:10]  # Limit to 10 keywords

    def _detect_rule_based_sentiment(self, message_content: str) -> str:
        """Detect sentiment using simple rules."""
        content_lower = message_content.lower()
        
        # Positive indicators
        positive_words = ["good", "great", "excellent", "positive", "bullish", "up", "rising", "gains"]
        # Negative indicators
        negative_words = ["bad", "poor", "terrible", "negative", "bearish", "down", "falling", "losses", "pressure"]
        
        positive_score = sum(1 for word in positive_words if word in content_lower)
        negative_score = sum(1 for word in negative_words if word in content_lower)
        
        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"

    def _detect_rule_based_intent(self, message_content: str) -> List[str]:
        """Detect intent signals using simple rules."""
        content_lower = message_content.lower()
        intents = []
        
        # Intent patterns
        if any(word in content_lower for word in ["need", "looking for", "want", "require"]):
            intents.append("information_seeking")
        
        if any(word in content_lower for word in ["buy", "sell", "trade", "invest"]):
            intents.append("trading_intent")
        
        if any(word in content_lower for word in ["analysis", "research", "report", "data"]):
            intents.append("research_request")
        
        if "?" in message_content:
            intents.append("question")
        
        return intents

    def _detect_rule_based_products(self, message_content: str) -> List[str]:
        """Detect product mentions using simple rules."""
        content_upper = message_content.upper()
        products = []
        
        # Currency pairs
        currency_patterns = ["USD/CAD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
        for pattern in currency_patterns:
            if pattern in content_upper:
                products.append(pattern)
        
        # Product types
        product_types = ["swap", "option", "forward", "bond", "equity", "derivative"]
        for product in product_types:
            if product.lower() in message_content.lower():
                products.append(product)
        
        return products

    def _add_fallback_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add minimal fallback features when processing fails."""
        record.update({
            "summary": "Feature extraction failed",
            "keywords": [],
            "sentiment": "unknown",
            "intent_signals": [],
            "product_mentions": [],
            "feature_extraction_timestamp": datetime.now().isoformat(),
            "feature_extraction_method": "fallback"
        })
        return record

    # LLM prompt templates
    def _get_summary_prompt(self) -> str:
        return """
        Summarize the following Bloomberg chat message in 1-2 sentences, focusing on the key business or financial topics discussed:

        Message: {message}

        Summary:
        """

    def _get_keywords_prompt(self) -> str:
        return """
        Extract the top 5 most important keywords or topics from this Bloomberg chat message. Focus on financial terms, products, sectors, and business topics.

        Message: {message}

        Keywords (comma-separated):
        """

    def _get_sentiment_prompt(self) -> str:
        return """
        Analyze the sentiment of this Bloomberg chat message. Respond with one word: positive, negative, or neutral.

        Message: {message}

        Sentiment:
        """

    def _get_intent_prompt(self) -> str:
        return """
        Identify the intent or purpose of this Bloomberg chat message. Common intents include: information_seeking, trading_intent, research_request, market_commentary, question, update.

        Message: {message}

        Intent signals (comma-separated):
        """

    def _get_products_prompt(self) -> str:
        return """
        Identify any financial products, instruments, or currency pairs mentioned in this Bloomberg chat message.

        Message: {message}

        Products mentioned (comma-separated):
        """

    # Helper methods for parsing LLM outputs
    def _clean_llm_output(self, output: str) -> str:
        """Clean and normalize LLM output."""
        if not output:
            return ""
        
        # Remove common LLM artifacts
        output = output.strip()
        output = output.replace("Summary:", "").replace("summary:", "")
        
        return output.strip()

    def _parse_keywords(self, output: str) -> List[str]:
        """Parse keywords from LLM output."""
        if not output:
            return []
        
        # Clean and split by commas
        keywords = [kw.strip() for kw in output.split(",")]
        # Remove empty strings and limit to 10
        keywords = [kw for kw in keywords if kw][:10]
        
        return keywords

    def _parse_sentiment(self, output: str) -> str:
        """Parse sentiment from LLM output."""
        if not output:
            return "unknown"
        
        output_lower = output.lower().strip()
        
        if "positive" in output_lower:
            return "positive"
        elif "negative" in output_lower:
            return "negative"
        elif "neutral" in output_lower:
            return "neutral"
        else:
            return "unknown"

    def _parse_intent_signals(self, output: str) -> List[str]:
        """Parse intent signals from LLM output."""
        if not output:
            return []
        
        # Split by commas and clean
        intents = [intent.strip() for intent in output.split(",")]
        intents = [intent for intent in intents if intent][:5]  # Limit to 5
        
        return intents

    def _parse_product_mentions(self, output: str) -> List[str]:
        """Parse product mentions from LLM output."""
        if not output:
            return []
        
        # Split by commas and clean
        products = [product.strip() for product in output.split(",")]
        products = [product for product in products if product][:10]  # Limit to 10
        
        return products 