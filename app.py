"""
KEITH Running Floor II Installation Assistant
Streamlit App
"""

import streamlit as st
from runningfloor_rag import RunningFloorRAG
import os

# Page config
st.set_page_config(
    page_title="KEITH Running Floor II Assistant",
    page_icon="🚛",
    layout="wide"
)

def get_api_key():
    """Get API key from Streamlit secrets or environment variable."""
    try:
        # Try Streamlit secrets first (for Streamlit Cloud)
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fall back to environment variable (for local development)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return api_key

def initialize_rag():
    """Initialize the RAG system with proper error handling."""
    api_key = get_api_key()
    
    if not api_key:
        st.error("❌ OPENAI_API_KEY not found!")
        st.info("""
        Please add your OpenAI API key:
        
        **For Streamlit Cloud:**
        1. Click 'Manage app' in the lower right
        2. Go to Settings → Secrets
        3. Add: `OPENAI_API_KEY = "sk-your-key-here"`
        
        **For Local Development:**
        - Set environment variable: `export OPENAI_API_KEY=sk-your-key-here`
        - Or create a `.streamlit/secrets.toml` file with the key
        """)
        st.stop()
    
    pdf_path = "keith_running_floor_ii_installation_manual.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        st.error(f"❌ PDF file not found: {pdf_path}")
        st.info("Please ensure the installation manual PDF is in the same directory as app.py")
        st.stop()
    
    try:
        rag = RunningFloorRAG(api_key=api_key, pdf_path=pdf_path)
        return rag
    except ValueError as e:
        st.error(f"❌ Error initializing RAG: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.stop()

def main():
    st.title("🚛 KEITH Running Floor II Installation Assistant")
    st.markdown("Ask questions about the Running Floor II installation manual")
    
    # Initialize RAG system (only once per session)
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Initializing RAG system... This may take a minute."):
            st.session_state.rag = initialize_rag()
            
            # Initialize the system (process PDF and generate embeddings)
            if not st.session_state.rag.initialize():
                st.error("❌ Failed to initialize RAG system")
                st.stop()
            
            st.session_state.initialized = True
    
    rag = st.session_state.rag
    
    # Show success message after initialization
    if st.session_state.get('initialized'):
        st.success("✅ System ready! Ask your questions below.")
        st.session_state.initialized = False  # Only show once
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask a Question")
        question = st.text_area(
            "Enter your question about the installation:",
            height=100,
            placeholder="e.g., How do I install the drive unit?"
        )
        
        ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Example Questions")
        examples = [
            "What are the main installation steps?",
            "How do I align the drive unit?",
            "What tools are needed?",
            "How do I install floor seals?",
            "What are the safety requirements?"
        ]
        
        for example in examples:
            if st.button(example, key=example, use_container_width=True):
                question = example
                ask_button = True
    
    # Process question
    if ask_button and question:
        with st.spinner("🤔 Searching the manual..."):
            result = rag.answer_question(question)
        
        if result['success']:
            st.markdown("---")
            st.subheader("💡 Answer")
            st.markdown(result['answer'])
            
            # Show sources in an expander
            with st.expander("📚 View Sources from Manual", expanded=False):
                for i, source in enumerate(result['sources'], 1):
                    st.markdown(f"**Source {i}: {source['section']}** (Page {source['page']})")
                    st.caption(source['preview'])
                    if i < len(result['sources']):
                        st.markdown("---")
        else:
            st.error("❌ Error")
            st.write(result['answer'])
    
    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.info("""
        This assistant helps you find information from the 
        **KEITH Running Floor II Installation Manual**.
        
        Ask questions in natural language and get answers based 
        on the official manual content.
        """)
        
        st.header("Tips")
        st.markdown("""
        - Be specific in your questions
        - Ask about installation steps, tools, or specifications
        - Check the sources to verify information
        - Reference page numbers for detailed procedures
        """)
        
        if st.session_state.get('rag'):
            st.header("System Status")
            num_chunks = len(st.session_state.rag.chunks)
            st.metric("Document Chunks", num_chunks)
            st.success("✅ System Active")

if __name__ == "__main__":
    main()
