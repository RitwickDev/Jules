"""
RFP Query Responder - Complete Python Implementation
=====================================================

A multi-agent AI system for automatically responding to bidder queries
during the RFP clarification process.

Author: Ritwick, Senior Digital Transformation Consultant
Project: PAN 2.0 - Income Tax Department
Version: 1.0
Date: December 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any, TypedDict, Annotated
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import operator
from functools import partial

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever

# LangGraph imports
from langgraph.graph import StateGraph, END

# Pydantic for data validation
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""

    # API Keys (should be set via environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model configuration
    LLM_MODEL = "gpt-4o"
    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_TEMPERATURE = 0

    # Vector store configuration
    CHROMA_PERSIST_DIR = "./chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Retrieval configuration
    TOP_K_RESULTS = 5

    # Domain mappings
    DOMAINS = [
        "technical",
        "functional",
        "commercial",
        "legal",
        "security",
        "infrastructure"
    ]

    # Collection mappings
    DOMAIN_COLLECTIONS = {
        "technical": [
            "rfp_vol1_technical",
            "technical_proposal",
            "bill_of_materials_software"
        ],
        "functional": [
            "rfp_vol1_functional",
            "clarifications",
            "frs_srs"
        ],
        "commercial": [
            "rfp_vol2_commercial",
            "bill_of_materials_pricing",
            "clarifications"
        ],
        "legal": [
            "rfp_vol3_contract",
            "clarifications"
        ],
        "security": [
            "rfp_vol1_security",
            "certin_directions_70b",
            "dpdp_act_2023"
        ],
        "infrastructure": [
            "rfp_vol1_infrastructure",
            "bill_of_materials_hardware",
            "technical_proposal"
        ]
    }


# =============================================================================
# DATA MODELS
# =============================================================================

class QueryDomain(str, Enum):
    """Enumeration of query domains"""
    TECHNICAL = "technical"
    FUNCTIONAL = "functional"
    COMMERCIAL = "commercial"
    LEGAL = "legal"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"


class QueryClassification(BaseModel):
    """Classification result for a query"""
    primary_domain: str = Field(
        description="Primary domain category"
    )
    secondary_domains: List[str] = Field(
        default=[],
        description="Additional domains if query spans multiple areas"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classification confidence score"
    )
    reasoning: str = Field(
        description="Explanation of classification decision"
    )


class AgentResponse(BaseModel):
    """Response from a specialist agent"""
    domain: str
    response_text: str
    citations: List[Dict[str, str]]
    chunk_count: int
    processing_time: float


class FinalResponse(BaseModel):
    """Final collated response"""
    query_id: str
    query_text: str
    response: str
    citations: List[str]
    cross_references: List[str]
    confidence: str
    contributing_agents: List[str]
    timestamp: str


# State definition for LangGraph
class QueryState(TypedDict):
    """State object for the query processing pipeline"""
    # Input
    query_id: str
    query_text: str
    query_volume: Optional[str]
    query_section: Optional[str]

    # Classification
    primary_domain: str
    secondary_domains: List[str]
    confidence_score: float
    is_multi_domain: bool

    # Routing
    assigned_agents: List[str]
    split_queries: List[Dict]

    # Processing
    agent_responses: Annotated[List[Dict], operator.add]
    retrieved_chunks: List[Dict]

    # Output
    final_response: str
    citations: List[str]
    cross_references: List[str]
    response_confidence: str
    contributing_agents: List[str]

    # Metadata
    processing_log: List[str]
    errors: List[str]


# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================

class RFPDocumentProcessor:
    """
    Processes RFP documents for ingestion into the vector store.
    Handles PDF, DOCX, and Excel files.
    """

    def __init__(self, persist_directory: str = Config.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        logger.info(f"Initialized DocumentProcessor with persist_dir: {persist_directory}")

    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        return loader.load()

    def process_and_store(
        self,
        file_path: str,
        collection_name: str,
        metadata: Optional[Dict] = None
    ) -> Chroma:
        """
        Process a document and store in vector database.

        Args:
            file_path: Path to the document file
            collection_name: Name for the vector store collection
            metadata: Additional metadata to attach to chunks

        Returns:
            Chroma vector store instance
        """
        logger.info(f"Processing document: {file_path}")

        # Load document
        documents = self.load_document(file_path)
        logger.info(f"Loaded {len(documents)} pages/sections")

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = f"{collection_name}_{i}"
            chunk.metadata['source_collection'] = collection_name
            chunk.metadata['source_file'] = file_path
            if metadata:
                chunk.metadata.update(metadata)

        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )

        logger.info(f"Stored in collection: {collection_name}")
        return vectorstore

    def process_batch(
        self,
        file_collection_pairs: List[Dict[str, str]]
    ) -> Dict[str, Chroma]:
        """
        Process multiple documents into their respective collections.

        Args:
            file_collection_pairs: List of dicts with 'file_path' and 'collection_name'

        Returns:
            Dictionary mapping collection names to vector stores
        """
        stores = {}
        for pair in file_collection_pairs:
            store = self.process_and_store(
                pair['file_path'],
                pair['collection_name'],
                pair.get('metadata')
            )
            stores[pair['collection_name']] = store
        return stores


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

class RFPKnowledgeBase:
    """
    Manages the RFP knowledge base with multiple collections.
    Provides domain-specific retrievers.
    """

    def __init__(self, persist_directory: str = Config.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.collections: Dict[str, Chroma] = {}
        logger.info("Initialized RFPKnowledgeBase")

    def load_collection(self, collection_name: str) -> Optional[Chroma]:
        """Load an existing collection from the persist directory"""
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.collections[collection_name] = vectorstore
            logger.info(f"Loaded collection: {collection_name}")
            return vectorstore
        except Exception as e:
            logger.warning(f"Could not load collection {collection_name}: {e}")
            return None

    def load_all_collections(self, collection_names: List[str]) -> None:
        """Load multiple collections"""
        for name in collection_names:
            self.load_collection(name)

    def get_retriever(
        self,
        collection_name: str,
        k: int = Config.TOP_K_RESULTS
    ):
        """Get a retriever for a specific collection"""
        if collection_name not in self.collections:
            self.load_collection(collection_name)

        if collection_name in self.collections:
            return self.collections[collection_name].as_retriever(
                search_kwargs={"k": k}
            )
        return None

    def get_domain_retriever(
        self,
        domain: str,
        k: int = Config.TOP_K_RESULTS
    ):
        """
        Get an ensemble retriever for a domain.
        Combines retrievers from all collections relevant to the domain.
        """
        domain_collections = Config.DOMAIN_COLLECTIONS.get(domain, [])

        retrievers = []
        weights = []

        for collection_name in domain_collections:
            retriever = self.get_retriever(collection_name, k)
            if retriever:
                retrievers.append(retriever)
                weights.append(1.0)

        if not retrievers:
            logger.warning(f"No retrievers available for domain: {domain}")
            return None

        return EnsembleRetriever(
            retrievers=retrievers,
            weights=weights
        )

    def search(
        self,
        query: str,
        domain: str,
        k: int = Config.TOP_K_RESULTS
    ) -> List[Document]:
        """Search for relevant documents in a domain"""
        retriever = self.get_domain_retriever(domain, k)
        if retriever:
            return retriever.invoke(query)
        return []


# =============================================================================
# QUERY CLASSIFIER
# =============================================================================

class QueryClassifier:
    """
    Classifies incoming queries into domain categories.
    Identifies single-domain and multi-domain queries.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert RFP query classifier for the PAN 2.0 Project.

Your task is to classify bidder queries into one or more domains:

DOMAINS:
- TECHNICAL: IT systems, software, APIs, databases, architecture, servers, storage
- FUNCTIONAL: PAN/TAN processes, user workflows, business rules, forms, application features
- COMMERCIAL: Pricing, costs, payment terms, commercial bid formats, schedules, unit rates
- LEGAL: Contract terms, liability, warranties, indemnity, termination, assignment clauses
- SECURITY: CERT-In compliance, DPDP Act, encryption, firewalls, certificates, audits
- INFRASTRUCTURE: Data centers, facilities, hardware, network connectivity, cabling, UPS

CLASSIFICATION RULES:
1. Identify the PRIMARY domain (most relevant)
2. List SECONDARY domains if query spans multiple areas
3. Provide confidence score (0.0 to 1.0)
4. Explain your reasoning briefly

{format_instructions}"""),
            ("human", "Classify this query:\n\n{query}")
        ])

    def classify(self, query: str) -> QueryClassification:
        """Classify a query into domain(s)"""
        try:
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })
            logger.info(f"Classified query as: {result.primary_domain}")
            return result
        except Exception as e:
            logger.error(f"Classification error: {e}")
            # Return default classification
            return QueryClassification(
                primary_domain="technical",
                secondary_domains=[],
                confidence=0.5,
                reasoning=f"Default classification due to error: {str(e)}"
            )


# =============================================================================
# SPECIALIST AGENTS
# =============================================================================

class SpecialistAgent:
    """
    Domain-specific agent for processing queries.
    Each agent has expertise in a particular domain.
    """

    SYSTEM_PROMPTS = {
        "technical": """You are a Technical RFP Expert for the PAN 2.0 Project (Income Tax Department).

Your expertise covers:
- IT Infrastructure (servers, storage, network)
- Software architecture and design
- API specifications and integrations
- Database requirements
- Non-functional requirements

RESPONSE GUIDELINES:
1. Provide accurate, specific answers based on RFP documentation
2. Always cite RFP sections with Volume number, Section number, and Page
3. Reference specific technical requirement IDs (e.g., NFR_XXX)
4. If information is derived/interpreted, clearly state this
5. Mention if existing clarifications address similar queries""",

        "functional": """You are a Functional RFP Expert for the PAN 2.0 Project (Income Tax Department).

Your expertise covers:
- PAN/TAN application processes
- User workflows and interfaces
- Functional requirement specifications
- Business rules and validations
- Helpdesk and grievance management

RESPONSE GUIDELINES:
1. Reference specific functional requirement IDs (e.g., PAN_PR_FR_XXX, CC_HD_FR_XXX)
2. Cite RFP Volume I Section 16 for functional requirements
3. Explain process flows clearly
4. Reference relevant clarifications if they address similar queries""",

        "commercial": """You are a Commercial RFP Expert for the PAN 2.0 Project (Income Tax Department).

Your expertise covers:
- Pricing and cost structures
- Payment terms and schedules
- Commercial bid formats (Schedules A through K)
- Unit rates and breakdowns
- Contract value calculations

RESPONSE GUIDELINES:
1. Reference specific Schedule names (A, B, C, etc.)
2. Cite RFP Volume II for commercial formats
3. Provide precise numerical references where applicable
4. Clarify cost components and calculation methodologies""",

        "legal": """You are a Legal RFP Expert for the PAN 2.0 Project (Income Tax Department).

Your expertise covers:
- Contract terms and conditions
- Liability and indemnity provisions
- Warranty requirements
- Termination and exit management
- Performance guarantees

RESPONSE GUIDELINES:
1. Reference specific clause numbers from RFP Volume III
2. Quote exact contractual language when relevant
3. Explain legal implications clearly
4. Note if response requires formal legal interpretation""",

        "security": """You are a Security RFP Expert for the PAN 2.0 Project (Income Tax Department).

Your expertise covers:
- CERT-In Directions 70B compliance
- DPDP Act 2023 requirements
- Security infrastructure (firewalls, SIEM, WAF, etc.)
- Encryption and key management
- Security certifications and audits

RESPONSE GUIDELINES:
1. Reference specific security requirement sections (17.3.2 - 17.3.5)
2. Cite CERT-In Directions where applicable
3. Reference ISO standards and compliance frameworks
4. Explain security implications clearly""",

        "infrastructure": """You are an Infrastructure RFP Expert for the PAN 2.0 Project (Income Tax Department).

Your expertise covers:
- Data Center requirements (DC, NDC)
- Facility setup and specifications
- Hardware components and specifications
- Network connectivity
- Physical infrastructure

RESPONSE GUIDELINES:
1. Reference specific infrastructure sections from RFP
2. Cite Bill of Materials for hardware specifications
3. Reference site-specific requirements (DC-1, DC-2, NDC-1, NDC-2)
4. Provide technical specifications where relevant"""
    }

    def __init__(
        self,
        domain: str,
        knowledge_base: RFPKnowledgeBase
    ):
        self.domain = domain
        self.knowledge_base = knowledge_base
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPTS.get(domain, self.SYSTEM_PROMPTS["technical"])),
            ("human", """Based on the following context from RFP documents:

{context}

Answer this query: {query}

Your response must include:
1. A clear, comprehensive answer
2. Specific citations (RFP Volume, Section, Page number)
3. Reference to any existing clarifications if applicable
4. Clear indication if information is derived/interpreted vs. directly from RFP""")
        ])

    def process(self, query: str) -> AgentResponse:
        """Process a query and return response with citations"""
        import time
        start_time = time.time()

        try:
            # Retrieve relevant documents
            docs = self.knowledge_base.search(query, self.domain)

            if not docs:
                return AgentResponse(
                    domain=self.domain,
                    response_text="No relevant information found in the knowledge base for this query.",
                    citations=[],
                    chunk_count=0,
                    processing_time=time.time() - start_time
                )

            # Build context from documents
            context = "\n\n---\n\n".join([
                f"Source: {doc.metadata.get('source_file', 'Unknown')}\n"
                f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                f"Content: {doc.page_content}"
                for doc in docs
            ])

            # Generate response
            chain = self.prompt | self.llm | StrOutputParser()
            response_text = chain.invoke({
                "context": context,
                "query": query
            })

            # Extract citations
            citations = [
                {
                    "source": doc.metadata.get('source_file', 'Unknown'),
                    "page": str(doc.metadata.get('page', 'Unknown')),
                    "collection": doc.metadata.get('source_collection', 'Unknown')
                }
                for doc in docs
            ]

            return AgentResponse(
                domain=self.domain,
                response_text=response_text,
                citations=citations,
                chunk_count=len(docs),
                processing_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return AgentResponse(
                domain=self.domain,
                response_text=f"Error processing query: {str(e)}",
                citations=[],
                chunk_count=0,
                processing_time=time.time() - start_time
            )


# =============================================================================
# QUERY SPLITTER
# =============================================================================

class QuerySplitter:
    """
    Splits multi-domain queries into domain-specific sub-queries.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query analyzer for the PAN 2.0 RFP project.

When given a complex query that spans multiple domains, split it into separate,
domain-specific questions that can be answered independently.

RULES:
1. Each split query should be self-contained and answerable independently
2. Preserve necessary context from the original query
3. Assign each split query to exactly one domain
4. Don't add information not present in the original query

Return your response as a JSON array with objects containing 'domain' and 'query' fields."""),
            ("human", """Split this multi-domain query into domain-specific questions:

Original Query: {query}
Identified Domains: {domains}

Return only the JSON array, no additional text.""")
        ])

    def split(self, query: str, domains: List[str]) -> List[Dict[str, str]]:
        """Split a query into domain-specific sub-queries"""
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "domains": ", ".join(domains)
            })

            # Parse JSON response
            split_queries = json.loads(result)
            logger.info(f"Split query into {len(split_queries)} sub-queries")
            return split_queries

        except Exception as e:
            logger.error(f"Query splitting error: {e}")
            # Fallback: use same query for all domains
            return [{"domain": d, "query": query} for d in domains]


# =============================================================================
# RESPONSE COLLATOR
# =============================================================================

class ResponseCollator:
    """
    Collates responses from multiple agents into a unified response.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response synthesizer for RFP query responses.

Your task is to merge responses from multiple specialist agents into a cohesive,
well-structured final response.

GUIDELINES:
1. Preserve all factual information from each agent
2. Remove redundant information
3. Organize the response logically
4. Maintain all citations
5. Ensure the response directly answers the original query"""),
            ("human", """Original Query: {query}

Agent Responses:
{agent_responses}

Create a unified response that:
1. Directly answers the query
2. Integrates information from all agents
3. Is well-structured and easy to read
4. Preserves all citations""")
        ])

    def collate(
        self,
        query: str,
        agent_responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """Collate multiple agent responses"""

        if len(agent_responses) == 1:
            # Single agent, no need to merge
            response = agent_responses[0]
            return {
                "response": response.response_text,
                "citations": self._format_citations(response.citations),
                "contributing_agents": [response.domain]
            }

        # Format agent responses for merging
        formatted_responses = "\n\n".join([
            f"--- {resp.domain.upper()} AGENT ---\n{resp.response_text}"
            for resp in agent_responses
        ])

        try:
            # Generate merged response
            chain = self.prompt | self.llm | StrOutputParser()
            merged_response = chain.invoke({
                "query": query,
                "agent_responses": formatted_responses
            })

            # Combine all citations
            all_citations = []
            for resp in agent_responses:
                all_citations.extend(resp.citations)

            return {
                "response": merged_response,
                "citations": self._format_citations(all_citations),
                "contributing_agents": [r.domain for r in agent_responses]
            }

        except Exception as e:
            logger.error(f"Collation error: {e}")
            # Fallback: concatenate responses
            return {
                "response": formatted_responses,
                "citations": self._format_citations(
                    [c for r in agent_responses for c in r.citations]
                ),
                "contributing_agents": [r.domain for r in agent_responses]
            }

    def _format_citations(self, citations: List[Dict]) -> List[str]:
        """Format citations as readable strings and deduplicate"""
        formatted = []
        seen = set()

        for citation in citations:
            source = citation.get('source', 'Unknown')
            page = citation.get('page', 'Unknown')
            key = f"{source}-{page}"

            if key not in seen:
                seen.add(key)
                formatted.append(f"{source}, Page {page}")

        return formatted


# =============================================================================
# CONSISTENCY CHECKER
# =============================================================================

class ConsistencyChecker:
    """
    Validates responses against existing clarifications for consistency.
    """

    def __init__(self, clarifications_retriever):
        self.retriever = clarifications_retriever
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance expert for RFP query responses.

Your task is to check if a generated response is consistent with existing
official clarifications. Look for:
1. Direct contradictions
2. Conflicting information
3. Inconsistent interpretations

Be thorough but don't flag minor differences in wording."""),
            ("human", """Query: {query}

Generated Response:
{response}

Existing Related Clarifications:
{clarifications}

Assess consistency:
1. Is the response consistent with existing clarifications? (YES/NO)
2. List any contradictions found (if any)
3. Should this response be flagged for human review? (YES/NO)
4. Confidence level: HIGH/MEDIUM/LOW""")
        ])

    def check(self, query: str, response: str) -> Dict[str, Any]:
        """Check response consistency with existing clarifications"""
        try:
            # Retrieve similar clarifications
            existing = self.retriever.invoke(query) if self.retriever else []

            clarifications_text = "\n\n".join([
                f"Clarification S.No. {doc.metadata.get('sno', 'N/A')}:\n{doc.page_content}"
                for doc in existing
            ]) if existing else "No related clarifications found."

            # Check consistency
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "response": response,
                "clarifications": clarifications_text
            })

            # Parse result
            is_consistent = "YES" in result.upper().split('\n')[0]
            needs_review = "YES" in result.upper() and "REVIEW" in result.upper()

            return {
                "is_consistent": is_consistent,
                "needs_review": needs_review,
                "validation_details": result,
                "related_clarifications": [
                    doc.metadata.get('sno') for doc in existing
                ] if existing else []
            }

        except Exception as e:
            logger.error(f"Consistency check error: {e}")
            return {
                "is_consistent": True,
                "needs_review": True,
                "validation_details": f"Error during validation: {str(e)}",
                "related_clarifications": []
            }


# =============================================================================
# GRAPH NODES
# =============================================================================

def classify_node(state: QueryState) -> QueryState:
    """
    Classifies the query and updates the state.
    """
    logger.info("Node: classify_node")
    query_text = state["query_text"]

    classifier = QueryClassifier()
    classification = classifier.classify(query_text)

    is_multi_domain = len(classification.secondary_domains) > 0
    assigned_agents = []
    if is_multi_domain:
        assigned_agents = [classification.primary_domain] + classification.secondary_domains
    else:
        assigned_agents = [classification.primary_domain]

    updates = {
        "primary_domain": classification.primary_domain,
        "secondary_domains": classification.secondary_domains,
        "confidence_score": classification.confidence,
        "is_multi_domain": is_multi_domain,
        "assigned_agents": assigned_agents,
    }
    return updates

def split_node(state: QueryState) -> QueryState:
    """
    Splits the query into domain-specific sub-queries.
    """
    logger.info("Node: split_node")
    query_text = state["query_text"]
    assigned_agents = state["assigned_agents"]

    splitter = QuerySplitter()
    split_queries = splitter.split(query_text, assigned_agents)

    return {
        "split_queries": split_queries,
    }

def agent_node(state: QueryState, agent: SpecialistAgent, name: str) -> QueryState:
    """
    Executes a specialist agent on the query.
    """
    logger.info(f"Node: agent_node ({name})")

    if state["is_multi_domain"]:
        # Find the query for this specific agent
        query_for_agent = next(
            (q["query"] for q in state["split_queries"] if q["domain"] == name),
            state["query_text"]  # Fallback to original query
        )
    else:
        query_for_agent = state["query_text"]

    result = agent.process(query_for_agent)

    updates = {
        "agent_responses": [result.model_dump()]
    }
    return updates

def collate_node(state: QueryState) -> QueryState:
    """
    Collates the responses from the specialist agents.
    """
    logger.info("Node: collate_node")
    query_text = state["query_text"]
    agent_responses = [AgentResponse(**r) for r in state["agent_responses"]]

    collator = ResponseCollator()
    collated_response = collator.collate(query_text, agent_responses)

    return {
        "final_response": collated_response["response"],
        "citations": collated_response["citations"],
        "contributing_agents": collated_response["contributing_agents"],
    }

def validate_node(state: QueryState, checker: ConsistencyChecker) -> QueryState:
    """
    Validates the final response for consistency.
    """
    logger.info("Node: validate_node")
    query_text = state["query_text"]
    final_response = state["final_response"]

    validation_result = checker.check(query_text, final_response)

    # Determine confidence level based on classification and validation
    if state["confidence_score"] >= 0.85 and validation_result["is_consistent"]:
        confidence = "HIGH"
    elif state["confidence_score"] >= 0.60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "cross_references": validation_result["related_clarifications"],
        "response_confidence": confidence,
    }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class RFPQueryResponderGraph:
    """
    Main orchestrator for the RFP Query Responder system.
    Builds and runs the LangGraph-based workflow.
    """
    def __init__(self, knowledge_base: RFPKnowledgeBase):
        self.knowledge_base = knowledge_base

        # Initialize components
        self.agents = {
            domain: SpecialistAgent(domain, knowledge_base)
            for domain in Config.DOMAINS
        }
        self.consistency_checker = ConsistencyChecker(
            knowledge_base.get_retriever("clarifications")
        )

    def build_graph(self):
        workflow = StateGraph(QueryState)

        # Add nodes
        workflow.add_node("classify", classify_node)
        workflow.add_node("split", split_node)

        for domain in Config.DOMAINS:
            workflow.add_node(
                f"agent_{domain}",
                partial(agent_node, agent=self.agents[domain], name=domain)
            )

        workflow.add_node("collate", collate_node)
        workflow.add_node(
            "validate",
            lambda state: validate_node(state, self.consistency_checker)
        )

        # Define the routing logic
        def route_to_agents(state: QueryState) -> str:
            """Routes to splitter or a specific agent based on classification."""
            if state["is_multi_domain"]:
                return "split"
            else:
                return f"agent_{state['primary_domain']}"

        def route_after_split(state: QueryState) -> List[str]:
            """Dynamically routes to all assigned agents after splitting."""
            return [f"agent_{d}" for d in state["assigned_agents"]]

        # Define edges
        workflow.set_entry_point("classify")

        # Conditional routing after classification
        workflow.add_conditional_edges(
            "classify",
            route_to_agents,
            {
                "split": "split",
                **{f"agent_{d}": f"agent_{d}" for d in Config.DOMAINS}
            }
        )

        # After splitting, fan out to all relevant agents
        workflow.add_conditional_edges("split", route_after_split)

        # After each agent runs, its response is aggregated. Once all are done, collate.
        for domain in Config.DOMAINS:
            workflow.add_edge(f"agent_{domain}", "collate")

        workflow.add_edge("collate", "validate")
        workflow.add_edge("validate", END)

        return workflow.compile()

    def run(self, query: str, query_id: str) -> FinalResponse:
        graph = self.build_graph()
        initial_state = QueryState(
            query_id=query_id,
            query_text=query,
            agent_responses=[],
        )
        final_state = graph.invoke(initial_state)

        return FinalResponse(
            query_id=query_id,
            query_text=query,
            response=final_state["final_response"],
            citations=final_state["citations"],
            cross_references=final_state["cross_references"],
            confidence=final_state["response_confidence"],
            contributing_agents=final_state["contributing_agents"],
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def format_response(response: FinalResponse) -> str:
    """Format a FinalResponse as readable text"""
    return f"""
{'='*70}
QUERY RESPONSE
{'='*70}

Query ID: {response.query_id}
Timestamp: {response.timestamp}
Confidence: {response.confidence}

QUERY:
{response.query_text}

{'─'*70}
RESPONSE:
{response.response}

{'─'*70}
CITATIONS:
{chr(10).join(f'• {c}' for c in response.citations) if response.citations else 'No citations'}

CROSS-REFERENCES:
{chr(10).join(f'• S.No. {r}' for r in response.cross_references) if response.cross_references else 'No related clarifications'}

CONTRIBUTING AGENTS: {', '.join(response.contributing_agents)}
{'='*70}
"""

def main():
    """Example usage of the RFP Query Responder"""

    # Initialize knowledge base
    kb = RFPKnowledgeBase()

    # Load collections (assuming they exist)
    collections_to_load = [
        "rfp_vol1_technical",
        "rfp_vol1_functional",
        "rfp_vol2_commercial",
        "rfp_vol3_contract",
        "clarifications"
    ]
    kb.load_all_collections(collections_to_load)

    # Initialize responder
    responder = RFPQueryResponderGraph(kb)

    # Example queries
    queries = [
        {
            "query_id": "Q001",
            "query_text": "What is the minimum EPS requirement for SIEM solution?"
        },
        {
            "query_id": "Q002",
            "query_text": "Please provide the unit rate breakdown for SSL Inspector, WAF, "
                         "Server Load Balancer and their security certification requirements."
        }
    ]

    # Process queries
    for q in queries:
        response = responder.run(q['query_text'], q['query_id'])
        print(format_response(response))


if __name__ == "__main__":
    main()
