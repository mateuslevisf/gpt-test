#!/usr/bin/env python3
"""
Retrieval System Diagnostic
Test and debug the RAG retrieval process step by step
"""

import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pdf_processor import PDFProcessor

load_dotenv()

class RetrievalDiagnostic:
    """Debug retrieval system step by step"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.client = OpenAI(api_key=self.api_key)
        self.pdf_processor = PDFProcessor(chunk_size=200, chunk_overlap=50)  # Much smaller chunks
        self.chunks = []
        self.embeddings = []
        self.full_text = ""

    def load_and_analyze_pdf(self, pdf_path):
        """Load PDF and show detailed analysis"""
        print(f"=== LOADING PDF: {pdf_path} ===")

        # Process PDF
        result = self.pdf_processor.process_pdf(pdf_path)
        self.full_text = result['full_text']
        self.chunks = result['chunks']

        print(f"✅ PDF processed successfully")
        print(f"📄 Pages: {result['stats']['num_pages']}")
        print(f"📝 Total characters: {result['stats']['char_count']}")
        print(f"🔢 Estimated tokens: {result['stats']['estimated_tokens']}")
        print(f"📦 Chunks created: {len(self.chunks)}")
        print()

        return result

    def show_chunks_sample(self, max_chunks=5):
        """Show sample of chunks to verify content"""
        print("=== SAMPLE CHUNKS ===")

        for i, chunk in enumerate(self.chunks[:max_chunks]):
            print(f"\n--- Chunk {i} (ID: {chunk['chunk_id']}) ---")
            print(f"Length: {chunk['char_count']} chars, ~{chunk['token_estimate']} tokens")
            print(f"Content preview: {chunk['text'][:300]}...")
            print(f"Content end: ...{chunk['text'][-100:]}")

        if len(self.chunks) > max_chunks:
            print(f"\n... and {len(self.chunks) - max_chunks} more chunks")
        print()

    def search_chunks_for_keyword(self, keyword):
        """Search chunks for specific keyword"""
        print(f"=== SEARCHING CHUNKS FOR: '{keyword}' ===")

        matches = []
        for i, chunk in enumerate(self.chunks):
            if keyword.lower() in chunk['text'].lower():
                matches.append((i, chunk))

        print(f"Found {len(matches)} chunks containing '{keyword}':")

        for i, (chunk_idx, chunk) in enumerate(matches[:3]):  # Show first 3 matches
            print(f"\n--- Match {i+1}: Chunk {chunk_idx} ---")
            # Find the keyword in context
            text = chunk['text']
            keyword_pos = text.lower().find(keyword.lower())
            start = max(0, keyword_pos - 100)
            end = min(len(text), keyword_pos + len(keyword) + 100)
            context = text[start:end]
            print(f"Context: ...{context}...")

        return matches

    def create_embeddings_with_progress(self):
        """Create embeddings and show progress"""
        print("=== CREATING EMBEDDINGS ===")

        self.embeddings = []
        failed_chunks = []

        for i, chunk in enumerate(self.chunks):
            print(f"Processing chunk {i+1}/{len(self.chunks)}", end="... ")

            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk['text']
                )
                embedding = response.data[0].embedding
                self.embeddings.append(embedding)
                print("✅")
            except Exception as e:
                print(f"❌ Error: {e}")
                self.embeddings.append([])
                failed_chunks.append(i)

        successful = len([e for e in self.embeddings if e])
        print(f"\n📊 Results:")
        print(f"  Successful embeddings: {successful}/{len(self.chunks)}")
        print(f"  Failed embeddings: {len(failed_chunks)}")
        if failed_chunks:
            print(f"  Failed chunk IDs: {failed_chunks}")
        print()

    def test_query_embedding(self, query):
        """Test query embedding creation"""
        print(f"=== TESTING QUERY EMBEDDING: '{query}' ===")

        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding
            print(f"✅ Query embedding created successfully")
            print(f"📐 Embedding dimensions: {len(query_embedding)}")
            return query_embedding
        except Exception as e:
            print(f"❌ Failed to create query embedding: {e}")
            return None

    def test_similarity_search(self, query, top_k=5):
        """Test similarity search step by step"""
        print(f"=== SIMILARITY SEARCH: '{query}' ===")

        # Get query embedding
        query_embedding = self.test_query_embedding(query)
        if not query_embedding:
            return []

        # Filter valid embeddings
        valid_embeddings = []
        valid_indices = []
        for i, embedding in enumerate(self.embeddings):
            if embedding:  # Not empty
                valid_embeddings.append(embedding)
                valid_indices.append(i)

        print(f"📊 Valid embeddings for comparison: {len(valid_embeddings)}")

        if not valid_embeddings:
            print("❌ No valid embeddings found!")
            return []

        # Calculate similarities
        similarities = cosine_similarity([query_embedding], valid_embeddings)[0]

        # Get top k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        print(f"\n🔍 Top {top_k} most similar chunks:")
        results = []

        for rank, idx in enumerate(top_indices, 1):
            original_idx = valid_indices[idx]
            chunk = self.chunks[original_idx]
            similarity = similarities[idx]

            print(f"\n--- Rank {rank} (Similarity: {similarity:.4f}) ---")
            print(f"Chunk ID: {original_idx}")
            print(f"Preview: {chunk['text'][:200]}...")

            results.append((chunk, similarity))

        return results

    def full_diagnostic(self, pdf_path, test_query):
        """Run complete diagnostic"""
        print("🔬 RETRIEVAL SYSTEM DIAGNOSTIC")
        print("=" * 50)

        # Step 1: Load PDF
        self.load_and_analyze_pdf(pdf_path)

        # Step 2: Show sample chunks
        self.show_chunks_sample()

        # Step 3: Search for keywords related to the query
        print("=== KEYWORD SEARCH ===")
        population_matches = self.search_chunks_for_keyword("população")
        if not population_matches:
            population_matches = self.search_chunks_for_keyword("population")
        if not population_matches:
            population_matches = self.search_chunks_for_keyword("hab")
        if not population_matches:
            population_matches = self.search_chunks_for_keyword("6 211")
        if not population_matches:
            population_matches = self.search_chunks_for_keyword("habitantes")

        # Step 4: Create embeddings
        self.create_embeddings_with_progress()

        # Step 5: Test similarity search
        results = self.test_similarity_search(test_query)

        # Step 6: Final analysis
        print("=== DIAGNOSTIC SUMMARY ===")
        print(f"✅ Chunks created: {len(self.chunks)}")
        print(f"✅ Embeddings created: {len([e for e in self.embeddings if e])}")
        print(f"✅ Similar chunks found: {len(results)}")

        if results:
            print(f"🎯 Best similarity score: {results[0][1]:.4f}")
            if results[0][1] < 0.5:
                print("⚠️  Low similarity scores - embeddings might not be capturing semantic meaning well")
        else:
            print("❌ No similar chunks found")

        return results

def main():
    """Interactive diagnostic"""
    try:
        diagnostic = RetrievalDiagnostic()

        print("🔬 Retrieval System Diagnostic Tool")
        print("Commands:")
        print("  full <pdf_path> <query>  - Run complete diagnostic")
        print("  load <pdf_path>          - Load PDF only")
        print("  chunks                   - Show chunk samples")
        print("  search <keyword>         - Search chunks for keyword")
        print("  embed                    - Create embeddings")
        print("  query <question>         - Test similarity search")
        print("  quit                     - Exit")
        print("=" * 50)

        while True:
            user_input = input("\n> ").strip()

            if user_input.lower() == 'quit':
                break

            elif user_input.lower().startswith('full '):
                parts = user_input[5:].strip().split(' ', 1)
                if len(parts) == 2:
                    pdf_path, query = parts
                    if os.path.exists(pdf_path):
                        diagnostic.full_diagnostic(pdf_path, query)
                    else:
                        print("PDF file not found")
                else:
                    print("Usage: full <pdf_path> <query>")

            elif user_input.lower().startswith('load '):
                pdf_path = user_input[5:].strip()
                if os.path.exists(pdf_path):
                    diagnostic.load_and_analyze_pdf(pdf_path)
                else:
                    print("PDF file not found")

            elif user_input.lower() == 'chunks':
                if diagnostic.chunks:
                    diagnostic.show_chunks_sample()
                else:
                    print("No PDF loaded")

            elif user_input.lower().startswith('search '):
                keyword = user_input[7:].strip()
                if diagnostic.chunks:
                    diagnostic.search_chunks_for_keyword(keyword)
                else:
                    print("No PDF loaded")

            elif user_input.lower() == 'embed':
                if diagnostic.chunks:
                    diagnostic.create_embeddings_with_progress()
                else:
                    print("No PDF loaded")

            elif user_input.lower().startswith('query '):
                query = user_input[6:].strip()
                if diagnostic.embeddings:
                    diagnostic.test_similarity_search(query)
                else:
                    print("No embeddings created")

            elif not user_input:
                continue

            else:
                print("Unknown command")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()