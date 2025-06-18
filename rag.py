#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System
Uses embeddings and similarity search for document retrieval
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import json
from pdf_processor import PDFProcessor

load_dotenv()

class RAGSystem:
    """Retrieval-Augmented Generation system using embeddings"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RAG system

        Args:
            api_key (str, optional): OpenAI API key
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.chunks = []
        self.embeddings = []
        self.metadata = {}
        self.pdf_processor = PDFProcessor(chunk_size=200, chunk_overlap=50)  # Smaller chunks for better retrieval

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Get embedding for text using OpenAI's embedding model

        Args:
            text (str): Text to embed
            model (str): Embedding model to use

        Returns:
            List[float]: Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return []

    def create_embeddings(self, chunks: List[Dict]) -> List[List[float]]:
        """
        Create embeddings for all chunks

        Args:
            chunks (List[Dict]): List of text chunks

        Returns:
            List[List[float]]: List of embedding vectors
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = []

        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"Processing chunk {i+1}/{len(chunks)}")

            embedding = self.get_embedding(chunk['text'])
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Failed to generate embedding for chunk {i}")
                embeddings.append([])  # Placeholder for failed embedding

        print(f"Generated {len([e for e in embeddings if e])} successful embeddings")
        return embeddings

    def find_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find most relevant chunks for a query using similarity search

        Args:
            query (str): User query
            top_k (int): Number of top chunks to return

        Returns:
            List[Tuple[Dict, float]]: List of (chunk, similarity_score) tuples
        """
        if not self.embeddings or not any(self.embeddings):
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        # Filter out empty embeddings
        valid_embeddings = []
        valid_indices = []
        for i, embedding in enumerate(self.embeddings):
            if embedding:
                valid_embeddings.append(embedding)
                valid_indices.append(i)

        if not valid_embeddings:
            return []

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], valid_embeddings)[0]

        # Get top k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_chunks = []
        for idx in top_indices:
            original_idx = valid_indices[idx]
            relevant_chunks.append((self.chunks[original_idx], similarities[idx]))

        return relevant_chunks

    def build_context(self, relevant_chunks: List[Tuple[Dict, float]],
                     max_context_length: int = 4000) -> str:
        """
        Build context string from relevant chunks

        Args:
            relevant_chunks (List[Tuple[Dict, float]]): Relevant chunks with scores
            max_context_length (int): Maximum context length

        Returns:
            str: Context string
        """
        context_parts = []
        current_length = 0

        for chunk, similarity in relevant_chunks:
            chunk_text = chunk['text']
            chunk_header = f"[Chunk {chunk['chunk_id']}, Similarity: {similarity:.3f}]"
            full_chunk = f"{chunk_header}\n{chunk_text}"

            if current_length + len(full_chunk) <= max_context_length:
                context_parts.append(full_chunk)
                current_length += len(full_chunk)
            else:
                break

        return "\n\n---\n\n".join(context_parts)

    def process_document(self, pdf_path: str) -> None:
        """
        Process document for RAG system

        Args:
            pdf_path (str): Path to PDF file
        """
        # Process PDF
        result = self.pdf_processor.process_pdf(pdf_path, clean_for_full_context=False)

        self.chunks = result['chunks']
        self.metadata = result['metadata']

        # Create embeddings
        self.embeddings = self.create_embeddings(self.chunks)

    def answer_question(self, question: str, top_k: int = 5,
                       max_context_length: int = 4000) -> Dict:
        """
        Answer a question using RAG approach

        Args:
            question (str): User question
            top_k (int): Number of chunks to retrieve
            max_context_length (int): Maximum context length

        Returns:
            Dict: Answer with metadata
        """
        if not self.chunks:
            return {
                'answer': "No document has been processed yet.",
                'method': 'RAG',
                'chunks_used': 0,
                'similarity_scores': [],
                'retrieved_excerpts': []
            }

        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(question, top_k)

        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'method': 'RAG',
                'chunks_used': 0,
                'similarity_scores': [],
                'retrieved_excerpts': []
            }

        # Build context
        context = self.build_context(relevant_chunks, max_context_length)

        # Store retrieved excerpts for display
        retrieved_excerpts = []
        for chunk, similarity in relevant_chunks:
            retrieved_excerpts.append({
                'text': chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
                'similarity': similarity,
                'chunk_id': chunk['chunk_id']
            })

        # Create prompt - more encouraging to answer with available info
        system_prompt = f"""You are an AI assistant that answers questions based on document excerpts.

Document: {self.metadata.get('title', 'Unknown')}
Method: RAG (Retrieval-Augmented Generation)

Instructions:
- Always attempt to answer using any relevant information from the excerpts
- If you find any numbers, facts, or related information, use it in your answer
- Even if the information is partial or indirect, provide what you can find
- If the excerpts mention the topic but lack specific details, explain what IS mentioned
- Be creative in connecting available information to answer the question
- Only say you cannot answer if the excerpts are completely unrelated to the question topic
- Format: "Based on the available excerpts: [your answer]. However, [mention any limitations]" """

        user_prompt = f"""Context excerpts:
{context}

Question: {question}

Answer based on the available excerpts:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content

            return {
                'answer': answer,
                'method': 'RAG',
                'chunks_used': len(relevant_chunks),
                'similarity_scores': [score for _, score in relevant_chunks],
                'context_length': len(context),
                'retrieved_excerpts': retrieved_excerpts
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'method': 'RAG',
                'chunks_used': len(relevant_chunks),
                'similarity_scores': [score for _, score in relevant_chunks],
                'retrieved_excerpts': retrieved_excerpts
            }

    def save_index(self, filename: str) -> None:
        """Save RAG index to file"""
        data = {
            'metadata': self.metadata,
            'chunks': self.chunks,
            'embeddings': self.embeddings
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"RAG index saved to {filename}")

    def load_index(self, filename: str) -> None:
        """Load RAG index from file"""
        with open(filename, 'r') as f:
            data = json.load(f)

        self.metadata = data['metadata']
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        print(f"RAG index loaded from {filename}")