#!/usr/bin/env python3
"""
PDF Processing Utilities
Shared module for PDF extraction, cleaning, and chunking
"""

import os
import PyPDF2
import re
from typing import List, Dict, Optional

class PDFProcessor:
    """Handles PDF text extraction, cleaning, and chunking operations"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor

        Args:
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata = {}

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            str: Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                # Store metadata
                self.metadata = {
                    'filename': os.path.basename(pdf_path),
                    'filepath': pdf_path,
                    'num_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', 'Unknown') if pdf_reader.metadata else 'Unknown',
                    'author': pdf_reader.metadata.get('/Author', 'Unknown') if pdf_reader.metadata else 'Unknown'
                }

                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {page_num + 1}]\n{page_text}"

                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean and preprocess text

        Args:
            text (str): Raw text
            preserve_structure (bool): Whether to preserve page markers and structure

        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
        text = text.strip()

        if not preserve_structure:
            # Remove page markers
            text = re.sub(r'\[Page \d+\]\s*', '', text)
            # Convert to single spacing
            text = re.sub(r'\n+', ' ', text)

        return text

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text (str): Text to estimate

        Returns:
            int: Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks

        Args:
            text (str): Text to chunk

        Returns:
            List[Dict]: List of text chunks with metadata
        """
        chunks = []
        words = text.split()

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'word_start': i,
                'word_end': i + len(chunk_words),
                'char_count': len(chunk_text),
                'token_estimate': self.estimate_tokens(chunk_text)
            })

        return chunks

    def fits_in_context_window(self, text: str, context_window_size: int = 120000) -> bool:
        """
        Check if text fits in context window

        Args:
            text (str): Text to check
            context_window_size (int): Context window size in tokens (default: 120k for GPT-4)

        Returns:
            bool: Whether text fits in context window
        """
        estimated_tokens = self.estimate_tokens(text)
        # Reserve space for system prompt, user question, and response
        available_tokens = context_window_size * 0.7  # Use 70% of context window

        return estimated_tokens <= available_tokens

    def get_document_stats(self, text: str) -> Dict:
        """
        Get statistics about the document

        Args:
            text (str): Document text

        Returns:
            Dict: Document statistics
        """
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'estimated_tokens': self.estimate_tokens(text),
            'fits_in_context': self.fits_in_context_window(text),
            'num_pages': self.metadata.get('num_pages', 0),
            'title': self.metadata.get('title', 'Unknown')
        }

    def process_pdf(self, pdf_path: str, clean_for_full_context: bool = False) -> Dict:
        """
        Process PDF and return both full text and chunks

        Args:
            pdf_path (str): Path to PDF file
            clean_for_full_context (bool): Whether to clean text for full context use

        Returns:
            Dict: Contains full_text, chunks, metadata, and stats
        """
        print(f"Processing PDF: {pdf_path}")

        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)

        # Clean text (different cleaning for different use cases)
        if clean_for_full_context:
            full_text = self.clean_text(raw_text, preserve_structure=True)
        else:
            full_text = self.clean_text(raw_text, preserve_structure=False)

        # Create chunks
        chunks = self.chunk_text(full_text)

        # Get statistics
        stats = self.get_document_stats(full_text)

        print(f"Extracted {stats['char_count']} characters from {stats['num_pages']} pages")
        print(f"Estimated {stats['estimated_tokens']} tokens")
        print(f"Fits in context window: {stats['fits_in_context']}")
        print(f"Created {len(chunks)} chunks")

        return {
            'full_text': full_text,
            'chunks': chunks,
            'metadata': self.metadata,
            'stats': stats
        }

# Convenience functions for backward compatibility
def extract_text_from_pdf(pdf_path: str) -> tuple:
    """Extract text and metadata from PDF"""
    processor = PDFProcessor()
    result = processor.process_pdf(pdf_path)
    return result['full_text'], result['metadata']

def clean_text(text: str, preserve_structure: bool = True) -> str:
    """Clean text using default processor"""
    processor = PDFProcessor()
    return processor.clean_text(text, preserve_structure)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """Chunk text using default processor"""
    processor = PDFProcessor(chunk_size, chunk_overlap)
    return processor.chunk_text(text)

def estimate_tokens(text: str) -> int:
    """Estimate token count"""
    processor = PDFProcessor()
    return processor.estimate_tokens(text)