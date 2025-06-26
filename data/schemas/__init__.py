"""
Data schemas for the recommendation system.
This module defines all data structures used across different connectors.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class PublicationType(str, Enum):
    """Publication types with their expiry rules."""
    MORNING_COMMENT = "morning_comment"
    WEEKLY_REPORT = "weekly_report"
    DEEP_DIVE = "deep_dive"
    DEFAULT = "default"


class PublicationData(BaseModel):
    """Schema for publication data."""
    id: str = Field(..., description="Unique publication identifier")
    title: str = Field(..., description="Publication title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    date: datetime = Field(..., description="Publication date")
    full_text: str = Field(..., description="Full text content")
    summary: Optional[str] = Field(None, description="LLM-generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    target_audience: Optional[str] = Field(None, description="Target audience profile")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    publication_type: PublicationType = Field(PublicationType.DEFAULT, description="Publication type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            # Try different date formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%B %d, %Y"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {v}")
        return v


class ChatMessage(BaseModel):
    """Schema for chat messages."""
    timestamp: datetime = Field(..., description="Message timestamp")
    participants: List[str] = Field(..., description="Chat participants")
    message_content: str = Field(..., description="Message content")
    client_id: Optional[str] = Field(None, description="Associated client ID")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID")
    summary: Optional[str] = Field(None, description="LLM-generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    sentiment: Optional[str] = Field(None, description="Sentiment analysis result")
    intent_signals: List[str] = Field(default_factory=list, description="Detected intent signals")
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            # Try different timestamp formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse timestamp: {v}")
        return v


class CRMEntry(BaseModel):
    """Schema for CRM data."""
    client_id: str = Field(..., description="Client identifier")
    timestamp: datetime = Field(..., description="Entry timestamp")
    entry_type: str = Field(..., description="Type of CRM entry (call, meeting, note)")
    content: str = Field(..., description="Entry content")
    summary: Optional[str] = Field(None, description="LLM-generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RFQData(BaseModel):
    """Schema for RFQ (Request for Quote) data."""
    client_id: str = Field(..., description="Client identifier")
    timestamp: datetime = Field(..., description="RFQ timestamp")
    product: str = Field(..., description="Financial product")
    currency_pair: Optional[str] = Field(None, description="Currency pair if applicable")
    notional: Optional[float] = Field(None, description="Notional amount")
    status: str = Field(..., description="RFQ status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClientKYC(BaseModel):
    """Schema for client KYC data."""
    client_id: str = Field(..., description="Client identifier")
    company_name: str = Field(..., description="Company name")
    language: str = Field("en", description="Preferred language")
    domicile: str = Field(..., description="Company domicile")
    sector: str = Field(..., description="Business sector")
    tier: Optional[str] = Field(None, description="Client tier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ReadershipData(BaseModel):
    """Schema for readership tracking data."""
    client_id: str = Field(..., description="Client identifier")
    publication_id: str = Field(..., description="Publication identifier")
    timestamp: datetime = Field(..., description="Read timestamp")
    engagement_type: str = Field(..., description="Type of engagement (view, click, download)")
    duration: Optional[int] = Field(None, description="Engagement duration in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClientProfile(BaseModel):
    """Comprehensive client profile schema."""
    client_id: str = Field(..., description="Client identifier")
    kyc_data: Optional[ClientKYC] = Field(None, description="KYC information")
    chat_features: List[ChatMessage] = Field(default_factory=list, description="Chat features")
    rfq_features: List[RFQData] = Field(default_factory=list, description="RFQ features")
    crm_features: List[CRMEntry] = Field(default_factory=list, description="CRM features")
    readership_features: List[ReadershipData] = Field(default_factory=list, description="Readership features")
    embedding: Optional[List[float]] = Field(None, description="Profile embedding vector")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class SourceEvidence(BaseModel):
    """Schema for source-specific evidence."""
    source: str = Field(..., description="Source name")
    evidence: str = Field(..., description="Evidence text")
    score: float = Field(..., ge=0, le=100, description="Source-specific score (0-100)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Recommendation(BaseModel):
    """Schema for a single recommendation."""
    publication: PublicationData = Field(..., description="Recommended publication")
    client_id: str = Field(..., description="Target client ID")
    aggregate_score: float = Field(..., ge=0, le=100, description="Aggregate recommendation score")
    relevance_summary: str = Field(..., description="Client-specific relevance explanation")
    source_evidence: List[SourceEvidence] = Field(..., description="Evidence from each source")
    timestamp: datetime = Field(default_factory=datetime.now, description="Recommendation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmailReport(BaseModel):
    """Schema for email report generation."""
    salesperson_name: str = Field(..., description="Target salesperson name")
    date: datetime = Field(default_factory=datetime.now, description="Report date")
    client_recommendations: Dict[str, List[Recommendation]] = Field(..., description="Recommendations by client")
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Email content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConnectorConfig(BaseModel):
    """Schema for connector configuration."""
    connector_name: str = Field(..., description="Connector name")
    enabled: bool = Field(True, description="Whether connector is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Connector-specific configuration")
    lookback_days: Optional[int] = Field(None, description="Lookback period in days")
    rate_limit: Optional[int] = Field(None, description="Rate limit for API calls")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PipelineConfig(BaseModel):
    """Schema for pipeline configuration."""
    recommendation_threshold: float = Field(65.0, ge=0, le=100, description="Minimum score threshold")
    max_recommendations_per_client: int = Field(5, ge=1, le=20, description="Maximum recommendations per client")
    scoring_weights: Dict[str, float] = Field(..., description="Scoring weights for different sources")
    embedding_model: str = Field("text-embedding-ada-002", description="Embedding model to use")
    llm_model: str = Field("gpt-4", description="LLM model for text generation")
    vector_search_k: int = Field(20, ge=1, le=100, description="Number of candidates to retrieve")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('scoring_weights')
    def validate_weights(cls, v):
        """Validate that scoring weights sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
        return v


class BusinessRules(BaseModel):
    """Schema for business rules configuration."""
    lookback_periods_days: Dict[str, int] = Field(..., description="Lookback periods by source")
    publication_expiry_days: Dict[str, int] = Field(..., description="Publication expiry by type")
    client_filters: Dict[str, Any] = Field(default_factory=dict, description="Client filtering rules")
    publication_filters: Dict[str, Any] = Field(default_factory=dict, description="Publication filtering rules")
    compliance_rules: Dict[str, Any] = Field(default_factory=dict, description="Compliance rules")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 