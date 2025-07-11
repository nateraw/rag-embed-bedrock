"""
RAG retrieval functions for downstream application.
Use these functions in your Streamlit app to perform retrieval-augmented generation.
"""

import json
import numpy as np
import boto3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""
    chunk_id: int
    text: str
    source_file: str
    page_number: Optional[int]
    similarity_score: float
    metadata: Dict


class RAGRetriever:
    """Handles retrieval operations for RAG."""
    
    def __init__(
        self,
        embeddings_dir: str = './embeddings',
        region_name: str = 'us-east-1',
        rerank: bool = True
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.region_name = region_name
        self.rerank = rerank
        
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        
        # Load embeddings and chunks
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embeddings and chunks from disk."""
        # Load chunks
        with open(self.embeddings_dir / 'chunks.json', 'r') as f:
            self.chunks = json.load(f)
        
        # Load embeddings
        self.embeddings = np.load(self.embeddings_dir / 'embeddings.npy')
        
        # Load metadata
        with open(self.embeddings_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Loaded {len(self.chunks)} chunks with embeddings")
    
    def create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for a query using Cohere with search_query input type."""
        body = json.dumps({
            "texts": [query],
            "input_type": "search_query",  # Important: use search_query for queries
            "embedding_types": ["float"]
        })
        
        response = self.bedrock.invoke_model(
            body=body,
            modelId='cohere.embed-english-v3',
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return np.array(response_body['embeddings']['float'][0])
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """Perform similarity search to find relevant chunks."""
        # Create query embedding
        query_embedding = self.create_query_embedding(query)
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                chunk = self.chunks[idx]
                result = SearchResult(
                    chunk_id=idx,
                    text=chunk['text'],
                    source_file=chunk['source_file'],
                    page_number=chunk.get('page_number'),
                    similarity_score=float(similarities[idx]),
                    metadata=chunk.get('metadata', {})
                )
                results.append(result)
        
        return results
    
    def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3
    ) -> List[SearchResult]:
        """Rerank results using Claude for better relevance."""
        if not results:
            return results
        
        # Create prompt for reranking
        passages = "\n\n".join([
            f"Passage {i+1} (from {r.source_file}):\n{r.text}"
            for i, r in enumerate(results)
        ])
        
        prompt = f"""Given the following query and passages, rank the passages by relevance to the query.
Return only the passage numbers in order of relevance (most relevant first).

Query: {query}

{passages}

Return your response as a comma-separated list of passage numbers (e.g., "3,1,2")."""
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = self.bedrock.invoke_model(
            body=body,
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        ranking_text = response_body['content'][0]['text'].strip()
        
        try:
            # Parse ranking
            ranking = [int(x.strip()) - 1 for x in ranking_text.split(',')]
            # Reorder results based on ranking
            reranked = [results[i] for i in ranking if i < len(results)]
            return reranked[:top_k]
        except:
            # If parsing fails, return original results
            logger.warning("Failed to parse reranking, returning original order")
            return results[:top_k]
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        rerank_top_k: int = 3,
        similarity_threshold: float = 0.0
    ) -> List[SearchResult]:
        """Main retrieval function combining search and optional reranking."""
        # Perform similarity search
        results = self.similarity_search(query, k, similarity_threshold)
        
        # Optionally rerank results
        if self.rerank and len(results) > 1:
            results = self.rerank_results(query, results, rerank_top_k)
        
        return results


class RAGGenerator:
    """Handles generation for RAG using Amazon Bedrock."""
    
    def __init__(self, region_name: str = 'us-east-1', model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0'):
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.model_id = model_id
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[SearchResult],
        system_prompt: Optional[str] = None,
        max_tokens: int = 1500
    ) -> str:
        """Generate answer using retrieved context."""
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source_info = f"Source: {chunk.source_file}"
            if chunk.page_number:
                source_info += f", Page {chunk.page_number}"
            context_parts.append(f"[{i+1}] {source_info}\n{chunk.text}")
        
        context = "\n\n".join(context_parts)
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always cite your sources by referencing the passage numbers in square brackets (e.g., [1], [2]).
If the context doesn't contain enough information to fully answer the question, say so."""
        
        # Build messages
        messages = [
            {
                "role": "user",
                "content": f"""Context:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based on the context provided above. Cite your sources."""
            }
        ]
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": 0.3
        })
        
        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']


class SimpleRAG:
    """Simple interface combining retrieval and generation."""
    
    def __init__(
        self,
        embeddings_dir: str = './embeddings',
        region_name: str = 'us-east-1',
        rerank: bool = True,
        model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0'
    ):
        self.retriever = RAGRetriever(embeddings_dir, region_name, rerank)
        self.generator = RAGGenerator(region_name, model_id)
    
    def query(
        self,
        question: str,
        k: int = 5,
        rerank_top_k: int = 3,
        similarity_threshold: float = 0.0,
        system_prompt: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, any]:
        """Perform full RAG pipeline: retrieve and generate."""
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            question,
            k=k,
            rerank_top_k=rerank_top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Generate answer
        answer = self.generator.generate_answer(
            question,
            chunks,
            system_prompt=system_prompt
        )
        
        result = {'answer': answer}
        
        if return_sources:
            result['sources'] = [
                {
                    'text': chunk.text,
                    'source': chunk.source_file,
                    'page': chunk.page_number,
                    'score': chunk.similarity_score
                }
                for chunk in chunks
            ]
        
        return result


# Example usage functions for Streamlit app
def load_rag_from_s3(
    bucket_name: str,
    embeddings_prefix: str,
    local_dir: str = './embeddings',
    region_name: str = 'us-east-1'
) -> SimpleRAG:
    """Load embeddings from S3 and initialize RAG."""
    s3 = boto3.client('s3', region_name=region_name)
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Download embedding files
    files_to_download = ['chunks.json', 'embeddings.npy', 'metadata.json']
    for file_name in files_to_download:
        s3_key = f"{embeddings_prefix}/{file_name}"
        local_file = local_path / file_name
        logger.info(f"Downloading s3://{bucket_name}/{s3_key}")
        s3.download_file(bucket_name, s3_key, str(local_file))
    
    # Initialize RAG
    return SimpleRAG(local_dir, region_name)


# Example Streamlit app code
def example_streamlit_usage():
    """Example of how to use these functions in a Streamlit app."""
    import streamlit as st
    
    # Initialize RAG (do this once, maybe cache it)
    @st.cache_resource
    def get_rag():
        # For local development
        return SimpleRAG('./embeddings')
        
        # For production (loading from S3)
        # return load_rag_from_s3('my-bucket', 'embeddings/')
    
    rag = get_rag()
    
    # Query interface
    st.title("Executive Meeting Q&A")
    question = st.text_input("Ask a question about the quarterly meetings:")
    
    if st.button("Search") and question:
        with st.spinner("Searching..."):
            result = rag.query(question, k=5, rerank_top_k=3)
        
        # Display answer
        st.write("### Answer")
        st.write(result['answer'])
        
        # Display sources
        if result.get('sources'):
            st.write("### Sources")
            for i, source in enumerate(result['sources']):
                with st.expander(f"Source {i+1}: {source['source']} (Score: {source['score']:.3f})"):
                    if source.get('page'):
                        st.write(f"Page: {source['page']}")
                    st.write(source['text'])


if __name__ == '__main__':
    # Example usage
    rag = SimpleRAG('./embeddings')
    result = rag.query("What were the main topics discussed in Q3?")
    print(result['answer'])
