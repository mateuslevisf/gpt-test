#!/usr/bin/env python3
"""
Comparison System for RAG vs Full Context approaches
Allows side-by-side comparison of both methods
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI
from rag import RAGSystem
from full_context import FullContextSystem
from pdf_processor import PDFProcessor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ComparisonSystem:
    """System to compare RAG vs Full Context approaches"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize comparison system

        Args:
            api_key (str, optional): OpenAI API key
        """
        self.rag_system = RAGSystem(api_key)
        self.full_context_system = FullContextSystem(api_key)
        self.pdf_processor = PDFProcessor()
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)

        self.document_loaded = False
        self.document_stats = {}
        self.comparison_history = []

    def direct_llm_answer(self, question: str) -> Dict:
        """
        Get answer from LLM without any document context

        Args:
            question (str): Question to ask

        Returns:
            Dict: Answer with metadata
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content

            return {
                'answer': answer,
                'method': 'Direct LLM',
                'uses_document': False,
                'estimated_tokens_used': len(question) // 4 + len(answer) // 4
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'method': 'Direct LLM',
                'uses_document': False,
                'error': str(e)
            }

    def load_document(self, pdf_path: str) -> Dict:
        """
        Load document into both systems

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Dict: Loading results for both systems
        """
        print("Loading document into both systems...")

        # Get document stats first
        result = self.pdf_processor.process_pdf(pdf_path)
        self.document_stats = result['stats']

        # Load into RAG system
        print("\n--- Loading into RAG System ---")
        try:
            self.rag_system.process_document(pdf_path)
            rag_loaded = True
            rag_error = None
        except Exception as e:
            rag_loaded = False
            rag_error = str(e)
            print(f"RAG loading failed: {e}")

        # Load into Full Context system
        print("\n--- Loading into Full Context System ---")
        try:
            full_result = self.full_context_system.process_document(pdf_path)
            full_loaded = True
            full_error = None
        except Exception as e:
            full_loaded = False
            full_error = str(e)
            full_result = {}
            print(f"Full Context loading failed: {e}")

        self.document_loaded = rag_loaded or full_loaded

        # Analysis
        analysis = self.full_context_system.compare_with_rag_capability()

        return {
            'document_stats': self.document_stats,
            'rag_system': {
                'loaded': rag_loaded,
                'error': rag_error,
                'chunks_created': len(self.rag_system.chunks) if rag_loaded else 0
            },
            'full_context_system': {
                'loaded': full_loaded,
                'error': full_error,
                'fits_in_context': full_result.get('fits_in_context', False)
            },
            'analysis': analysis
        }

    def compare_answers(self, question: str, rag_params: Dict = None,
                       full_context_params: Dict = None, include_direct_llm: bool = True) -> Dict:
        """
        Get answers from all systems and compare

        Args:
            question (str): Question to ask all systems
            rag_params (Dict, optional): Parameters for RAG system
            full_context_params (Dict, optional): Parameters for Full Context system
            include_direct_llm (bool): Whether to include direct LLM answer

        Returns:
            Dict: Comparison results
        """
        rag_params = rag_params or {}
        full_context_params = full_context_params or {}

        print(f"\nComparing answers for: {question}")
        print("=" * 60)

        # Get RAG answer
        print("Getting RAG answer...")
        rag_result = self.rag_system.answer_question(question, **rag_params) if self.document_loaded else {
            'answer': 'No document loaded', 'method': 'RAG', 'chunks_used': 0, 'similarity_scores': []
        }

        # Get Full Context answer (if possible)
        print("Getting Full Context answer...")
        full_context_result = self.full_context_system.answer_question(question, **full_context_params) if self.document_loaded else {
            'answer': 'No document loaded', 'method': 'Full Context', 'fits_in_context': False
        }

        # Get Direct LLM answer
        direct_llm_result = None
        if include_direct_llm:
            print("Getting Direct LLM answer...")
            direct_llm_result = self.direct_llm_answer(question)

        # Create comparison
        comparison = {
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'rag_result': rag_result,
            'full_context_result': full_context_result,
            'direct_llm_result': direct_llm_result,
            'document_stats': self.document_stats if self.document_loaded else {}
        }

        # Add to history
        self.comparison_history.append(comparison)

        return comparison

    def print_comparison(self, comparison: Dict) -> None:
        """
        Print formatted comparison results

        Args:
            comparison (Dict): Comparison results
        """
        print(f"\nQUESTION: {comparison['question']}")
        print("=" * 80)

        # RAG Results
        print("\n🔍 RAG SYSTEM ANSWER:")
        print("-" * 40)
        rag_result = comparison['rag_result']
        print(f"Answer: {rag_result['answer']}")
        print(f"Chunks used: {rag_result.get('chunks_used', 0)}")
        if 'similarity_scores' in rag_result and rag_result['similarity_scores']:
            scores = [f"{s:.3f}" for s in rag_result['similarity_scores']]
            print(f"Similarity scores: {', '.join(scores)}")

        # Show retrieved excerpts
        if 'retrieved_excerpts' in rag_result and rag_result['retrieved_excerpts']:
            print("\n📋 Retrieved Excerpts:")
            for i, excerpt in enumerate(rag_result['retrieved_excerpts'], 1):
                print(f"  Excerpt {i} (Similarity: {excerpt['similarity']:.3f}):")
                print(f"    {excerpt['text']}")
                print()
        else:
            print(f"\n📋 Debug: retrieved_excerpts key exists: {'retrieved_excerpts' in rag_result}")
            if 'retrieved_excerpts' in rag_result:
                print(f"    retrieved_excerpts content: {rag_result['retrieved_excerpts']}")
            print("    (This suggests the RAG system isn't populating the excerpts field)")

        # Full Context Results
        print("\n📄 FULL CONTEXT SYSTEM ANSWER:")
        print("-" * 40)
        full_result = comparison['full_context_result']
        print(f"Answer: {full_result['answer']}")
        if 'fits_in_context' in full_result:
            print(f"Fits in context: {full_result['fits_in_context']}")
        if 'estimated_tokens_used' in full_result:
            print(f"Estimated tokens used: {full_result['estimated_tokens_used']}")

        # Direct LLM Results
        if comparison.get('direct_llm_result'):
            print("\n🤖 DIRECT LLM ANSWER:")
            print("-" * 40)
            direct_result = comparison['direct_llm_result']
            print(f"Answer: {direct_result['answer']}")
            print(f"Uses document: {direct_result.get('uses_document', False)}")
            if 'estimated_tokens_used' in direct_result:
                print(f"Estimated tokens used: {direct_result['estimated_tokens_used']}")

        # Analysis
        print("\n📊 COMPARISON ANALYSIS:")
        print("-" * 40)

        if full_result.get('fits_in_context', False):
            print("✅ Full Context: Complete document available")
            print("🔍 RAG: Retrieval-based (partial document)")
        else:
            print("❌ Full Context: Document too large")
            print("✅ RAG: Only document-based option")

        if comparison.get('direct_llm_result'):
            print("🤖 Direct LLM: No document context (baseline)")

        if 'document_stats' in comparison and comparison['document_stats']:
            print(f"\nDocument tokens: {comparison['document_stats'].get('estimated_tokens', 'Unknown')}")
            print(f"Document pages: {comparison['document_stats'].get('num_pages', 'Unknown')}")

    def run_batch_comparison(self, questions: List[str]) -> List[Dict]:
        """
        Run comparison on multiple questions

        Args:
            questions (List[str]): List of questions to compare

        Returns:
            List[Dict]: List of comparison results
        """
        results = []

        for i, question in enumerate(questions, 1):
            print(f"\n{'='*20} Question {i}/{len(questions)} {'='*20}")
            comparison = self.compare_answers(question)
            self.print_comparison(comparison)
            results.append(comparison)

        return results

    def save_comparison_results(self, filename: str) -> None:
        """
        Save all comparison results to file

        Args:
            filename (str): Output filename
        """
        data = {
            'document_stats': self.document_stats,
            'comparison_history': self.comparison_history,
            'summary': {
                'total_comparisons': len(self.comparison_history),
                'document_fits_in_context': self.document_stats.get('fits_in_context', False)
            }
        }

        full_filename = os.path.join('comparisons', filename)
        with open(full_filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Comparison results saved to comparisons/{filename}")

    def analyze_differences(self) -> Dict:
        """
        Analyze differences between RAG and Full Context answers

        Returns:
            Dict: Analysis of differences
        """
        if not self.comparison_history:
            return {'error': 'No comparisons performed yet'}

        analysis = {
            'total_comparisons': len(self.comparison_history),
            'full_context_available': 0,
            'rag_only': 0,
            'both_answered': 0,
            'common_patterns': []
        }

        for comp in self.comparison_history:
            full_result = comp['full_context_result']

            if full_result.get('fits_in_context', False):
                analysis['full_context_available'] += 1
                if 'error' not in full_result['answer'].lower():
                    analysis['both_answered'] += 1
            else:
                analysis['rag_only'] += 1

        # Add insights
        if analysis['both_answered'] > 0:
            analysis['common_patterns'].append("Both systems could answer some questions")

        if analysis['rag_only'] > 0:
            analysis['common_patterns'].append("Some documents required RAG due to size")

        return analysis

def main():
    """Example usage of the comparison system"""
    try:
        # Initialize comparison system
        comp_system = ComparisonSystem()

        print("RAG vs Full Context Comparison System")
        print("Commands:")
        print("  load <pdf_path>           - Load PDF into document-based systems")
        print("  compare <question>        - Compare answers from all three systems")
        print("  direct <question>         - Test direct LLM answer only")
        print("  batch <q1>|<q2>|<q3>     - Compare multiple questions (separated by |)")
        print("  analyze                   - Analyze differences between systems")
        print("  save <filename>           - Save comparison results")
        print("  quit                      - Exit")
        print("=" * 60)

        while True:
            user_input = input("\n> ").strip()

            if user_input.lower() == 'quit':
                break

            elif user_input.lower().startswith('load '):
                pdf_path = user_input[5:].strip()
                if os.path.exists(pdf_path):
                    try:
                        result = comp_system.load_document(pdf_path)
                        print("\n📊 LOADING RESULTS:")
                        print(f"Document: {result['document_stats']['title']}")
                        print(f"Pages: {result['document_stats']['num_pages']}")
                        print(f"Estimated tokens: {result['document_stats']['estimated_tokens']}")
                        print(f"Fits in context: {result['document_stats']['fits_in_context']}")
                        print(f"RAG loaded: {result['rag_system']['loaded']}")
                        print(f"Full Context loaded: {result['full_context_system']['loaded']}")
                        print(f"Recommended approach: {result['analysis']['recommended_approach']}")
                    except Exception as e:
                        print(f"Error loading PDF: {e}")
                else:
                    print("PDF file not found.")

            elif user_input.lower().startswith('compare '):
                question = user_input[8:].strip()
                if question:
                    comparison = comp_system.compare_answers(question)
                    comp_system.print_comparison(comparison)
                else:
                    print("Please provide a question.")

            elif user_input.lower().startswith('direct '):
                question = user_input[7:].strip()
                if question:
                    result = comp_system.direct_llm_answer(question)
                    print(f"\n🤖 DIRECT LLM ANSWER:")
                    print("-" * 40)
                    print(f"Answer: {result['answer']}")
                    print(f"Uses document: {result.get('uses_document', False)}")
                    if 'estimated_tokens_used' in result:
                        print(f"Estimated tokens used: {result['estimated_tokens_used']}")
                else:
                    print("Please provide a question.")

            elif user_input.lower().startswith('batch '):
                questions_str = user_input[6:].strip()
                questions = [q.strip() for q in questions_str.split('|') if q.strip()]
                if questions:
                    comp_system.run_batch_comparison(questions)
                else:
                    print("Please provide questions separated by |")

            elif user_input.lower() == 'analyze':
                analysis = comp_system.analyze_differences()
                print("\n📈 ANALYSIS RESULTS:")
                print(f"Total comparisons: {analysis['total_comparisons']}")
                print(f"Full context available: {analysis['full_context_available']}")
                print(f"RAG only: {analysis['rag_only']}")
                print(f"Both systems answered: {analysis['both_answered']}")
                for pattern in analysis['common_patterns']:
                    print(f"- {pattern}")

            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                if filename:
                    comp_system.save_comparison_results(filename)
                else:
                    print("Please provide a filename.")

            elif not user_input:
                continue

            else:
                print("Unknown command. Type 'quit' to exit.")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Make sure your OpenAI API key is set in the .env file")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()