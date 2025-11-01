"""
KEITH Running Floor II Installation Manual RAG System
=====================================================

A production-ready RAG system for the KEITH Running Floor II Drive Installation Manual.
Provides intelligent question-answering about installation procedures, specifications,
and technical guidance.

Author: Built with Claude
Version: 1.0.0
"""

import os
from typing import List, Dict, Optional
import openai
from pypdf import PdfReader
import numpy as np
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a semantic chunk of the installation manual."""
    content: str
    page_num: int
    section: str
    chunk_id: str
    embedding: Optional[np.ndarray] = None


class RunningFloorRAG:
    """
    RAG system for KEITH Running Floor II Installation Manual.
    
    This system handles:
    - PDF processing and semantic chunking
    - Embedding generation with OpenAI
    - Similarity search for relevant context
    - GPT-4 powered question answering
    """
    
    def __init__(self, api_key: str, pdf_path: str):
        """
        Initialize the RAG system.
        
        Args:
            api_key: OpenAI API key
            pdf_path: keith_running_floor_ii_installation_manual.pdf
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.pdf_path = pdf_path
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def process_document(self) -> None:
        """Extract and chunk the installation manual into semantic sections."""
        print("📄 Processing KEITH Running Floor II Installation Manual...")
        
        reader = PdfReader(self.pdf_path)
        total_pages = len(reader.pages)
        
        # Define semantic sections based on the manual structure
        sections = {
            "introduction": (1, 1),
            "trailer_prep": (2, 8),
            "subdeck": (9, 16),
            "drive_unit": (16, 21),
            "flooring": (22, 35),
            "hydraulic": (36, 37),
            "miscellaneous": (38, 41),
            "appendix": (42, 55)
        }
        
        chunk_id = 0
        
        for section_name, (start_page, end_page) in sections.items():
            section_text = ""
            
            for page_num in range(start_page - 1, min(end_page, total_pages)):
                page = reader.pages[page_num]
                section_text += page.extract_text() + "\n\n"
            
            # Create semantic chunks (split long sections)
            if len(section_text) > 3000:
                # Split into smaller chunks for better retrieval
                words = section_text.split()
                chunk_size = 1500
                overlap = 200
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    if len(chunk_text) > 100:  # Minimum chunk size
                        chunk = DocumentChunk(
                            content=chunk_text,
                            page_num=start_page + (i // chunk_size),
                            section=section_name.replace("_", " ").title(),
                            chunk_id=f"chunk_{chunk_id}"
                        )
                        self.chunks.append(chunk)
                        chunk_id += 1
            else:
                # Keep as single chunk
                chunk = DocumentChunk(
                    content=section_text,
                    page_num=start_page,
                    section=section_name.replace("_", " ").title(),
                    chunk_id=f"chunk_{chunk_id}"
                )
                self.chunks.append(chunk)
                chunk_id += 1
        
        print(f"✅ Created {len(self.chunks)} semantic chunks from {total_pages} pages")
    
    def generate_embeddings(self) -> None:
        """Generate embeddings for all document chunks using OpenAI."""
        print("🔄 Generating embeddings...")
        
        embeddings_list = []
        
        for i, chunk in enumerate(self.chunks):
            try:
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk.content
                )
                embedding = response.data[0].embedding
                chunk.embedding = np.array(embedding)
                embeddings_list.append(embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{len(self.chunks)} embeddings")
                    
            except Exception as e:
                print(f"⚠️ Error generating embedding for chunk {i}: {e}")
                continue
        
        self.embeddings = np.array(embeddings_list)
        print(f"✅ Generated {len(embeddings_list)} embeddings")
    
    def find_relevant_chunks(self, query: str, top_k: int = 4) -> List[DocumentChunk]:
        """
        Find the most relevant chunks for a given query.
        
        Args:
            query: User's question
            top_k: Number of chunks to return
            
        Returns:
            List of most relevant DocumentChunks
        """
        # Generate query embedding
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Calculate cosine similarity
        similarities = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                similarity = np.dot(query_embedding, chunk.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                )
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in similarities[:top_k]]
    
    def answer_question(self, question: str) -> Dict[str, any]:
        """
        Answer a question using the RAG system.
        
        Args:
            question: User's question about the installation manual
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Find relevant context
        relevant_chunks = self.find_relevant_chunks(question, top_k=4)
        
        # Build context from relevant chunks
        context = "\n\n---\n\n".join([
            f"Section: {chunk.section} (Page {chunk.page_num})\n{chunk.content}"
            for chunk in relevant_chunks
        ])
        
        # Create prompt for GPT-4
        system_prompt = """You are a technical expert assistant for KEITH Manufacturing Company, 
specializing in the Running Floor II® unloading system installation. You provide clear, 
accurate, and professional guidance based on the official installation manual.

When answering questions:
- Be precise and reference specific sections when relevant
- Include safety warnings where applicable
- Provide step-by-step instructions when needed
- Use technical terminology correctly
- Mention page numbers when citing specific procedures
- If information isn't in the manual, say so clearly"""

        user_prompt = f"""Based on the following sections from the KEITH Running Floor II Installation Manual, 
please answer this question:

Question: {question}

Relevant Manual Sections:
{context}

Please provide a clear, professional answer based on the manual content above."""

        # Get answer from GPT-4
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "section": chunk.section,
                        "page": chunk.page_num,
                        "preview": chunk.content[:200] + "..."
                    }
                    for chunk in relevant_chunks
                ],
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def initialize(self) -> bool:
        """
        Initialize the complete RAG system.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.process_document()
            self.generate_embeddings()
            print("✅ KEITH Running Floor II RAG system ready!")
            return True
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False


def main():
    """Example usage of the Running Floor RAG system."""
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    PDF_PATH = "/mnt/user-data/uploads/keith_running_floor_ii_installation_manual.pdf"
    
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Initialize RAG system
    rag = RunningFloorRAG(api_key=API_KEY, pdf_path=PDF_PATH)
    
    if not rag.initialize():
        print("❌ Failed to initialize RAG system")
        return
    
    # Example questions
    test_questions = [
        "What are the main steps for installing the drive unit?",
        "How do I align the drive unit properly?",
        "What tools are needed for installation?",
        "How do I install the floor seals?"
    ]
    
    print("\n" + "="*60)
    print("Testing KEITH Running Floor II RAG System")
    print("="*60 + "\n")
    
    for question in test_questions:
        print(f"❓ Question: {question}")
        result = rag.answer_question(question)
        print(f"\n💡 Answer:\n{result['answer']}\n")
        print(f"📚 Sources: {len(result['sources'])} sections referenced")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
