#!/usr/bin/env python3
"""
Full Context System
Uses entire document in context window when possible
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Optional
from pdf_processor import PDFProcessor

load_dotenv()

class FullContextSystem:
    """Full context system that includes entire document in prompt"""

    def __init__(self, api_key: Optional[str] = None, context_window_size: int = 120000):
        """
        Initialize full context system

        Args:
            api_key (str, optional): OpenAI API key
            context_window_size (int): Context window size in tokens
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.context_window_size = context_window_size
        self.full_text = ""
        self.metadata = {}
        self.pdf_processor = PDFProcessor()

    def process_document(self, pdf_path: str) -> Dict:
        """
        Process document for full context system

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Dict: Processing result with fit status
        """
        # Process PDF
        result = self.pdf_processor.process_pdf(pdf_path, clean_for_full_context=True)

        self.full_text = result['full_text']
        self.metadata = result['metadata']
        self.stats = result['stats']

        return {
            'fits_in_context': result['stats']['fits_in_context'],
            'estimated_tokens': result['stats']['estimated_tokens'],
            'available_tokens': int(self.context_window_size * 0.7),
            'stats': result['stats']
        }

    def answer_question(self, question: str, include_page_refs: bool = True) -> Dict:
        """
        Answer a question using full context approach

        Args:
            question (str): User question
            include_page_refs (bool): Whether to include page references

        Returns:
            Dict: Answer with metadata
        """
        if not self.full_text:
            return {
                'answer': "No document has been processed yet.",
                'method': 'Full Context',
                'fits_in_context': False,
                'total_tokens_used': 0
            }

        # Check if document fits in context window
        fits_in_context = self.pdf_processor.fits_in_context_window(
            self.full_text, self.context_window_size
        )

        if not fits_in_context:
            return {
                'answer': "Document is too large for full context processing. Consider using RAG approach.",
                'method': 'Full Context',
                'fits_in_context': False,
                'estimated_tokens': self.pdf_processor.estimate_tokens(self.full_text),
                'max_tokens': int(self.context_window_size * 0.7)
            }

        # Create system prompt
        system_prompt = f"""You are an AI assistant that answers questions based on the complete document provided.

Document: {self.metadata.get('title', 'Unknown')}
Method: Full Context (entire document available)
Pages: {self.metadata.get('num_pages', 'Unknown')}

Instructions:
- You have access to the COMPLETE document
- Answer comprehensively using any relevant information
- You can make connections across different parts of the document
- If asked about structure, organization, or relationships, you can see the full context"""

        if include_page_refs:
            system_prompt += "\n- When referencing specific information, mention the page number if available"

        # Prepare the full document context
        user_prompt = f"""Complete Document:
{self.full_text}

Question: {question}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content

            # Estimate tokens used
            total_context = system_prompt + user_prompt
            estimated_tokens_used = self.pdf_processor.estimate_tokens(total_context)

            return {
                'answer': answer,
                'method': 'Full Context',
                'fits_in_context': True,
                'estimated_tokens_used': estimated_tokens_used,
                'document_stats': self.stats
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'method': 'Full Context',
                'fits_in_context': fits_in_context,
                'error': str(e)
            }

    def get_document_summary(self) -> Dict:
        """
        Get a summary of the processed document

        Returns:
            Dict: Document summary and statistics
        """
        if not self.full_text:
            return {'error': 'No document processed'}

        summary_prompt = f"""Provide a comprehensive summary of this document including:
1. Main topic/subject
2. Key points or sections
3. Document structure
4. Important findings or conclusions

Document: {self.full_text[:2000]}{'...' if len(self.full_text) > 2000 else ''}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates document summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return {
                'summary': response.choices[0].message.content,
                'stats': self.stats,
                'metadata': self.metadata
            }

        except Exception as e:
            return {'error': f"Error generating summary: {str(e)}"}

    def compare_with_rag_capability(self) -> Dict:
        """
        Analyze document's suitability for full context vs RAG

        Returns:
            Dict: Analysis of which approach would be better
        """
        if not self.full_text:
            return {'error': 'No document processed'}

        stats = self.stats
        fits = stats['fits_in_context']
        tokens = stats['estimated_tokens']

        analysis = {
            'fits_in_context': fits,
            'estimated_tokens': tokens,
            'recommended_approach': 'Full Context' if fits else 'RAG',
            'reasoning': []
        }

        if fits:
            analysis['reasoning'].append("Document fits in context window")
            analysis['reasoning'].append("Full context allows complete document understanding")
            analysis['reasoning'].append("No risk of missing relevant information")
        else:
            analysis['reasoning'].append("Document exceeds context window")
            analysis['reasoning'].append("RAG required for processing")
            analysis['reasoning'].append(f"Document is {tokens - int(self.context_window_size * 0.7):.0f} tokens over limit")

        return analysis