"""
KEITH Running Floor II Installation Assistant
==============================================

Professional web interface for the KEITH Running Floor II RAG system.
Built with Streamlit for easy deployment and user interaction.
"""

import streamlit as st
import os
from runningfloor_rag import RunningFloorRAG


# Page configuration
st.set_page_config(
    page_title="KEITH Running Floor II Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for KEITH branding
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #C8102E;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B6B6B;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #C8102E;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E5E5E5;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .keith-badge {
        background-color: #C8102E;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #C8102E;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #A00D24;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """Initialize and cache the RAG system."""
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    pdf_path = "/mnt/user-data/uploads/keith_running_floor_ii_installation_manual.pdf"
    
    if not api_key:
        st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return None
    
    rag = RunningFloorRAG(api_key=api_key, pdf_path=pdf_path)
    
    with st.spinner("🔄 Initializing KEITH Running Floor II Assistant..."):
        if rag.initialize():
            return rag
        else:
            st.error("❌ Failed to initialize RAG system")
            return None


def main():
    """Main application logic."""
    
    # Header
    st.markdown('<div class="main-header">🔧 KEITH Running Floor II Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Professional Installation Guidance System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/C8102E/FFFFFF?text=KEITH+MFG", use_container_width=True)
        
        st.markdown("### 📋 About This System")
        st.info("""
        This AI assistant provides expert guidance on installing the **KEITH Running Floor II® Drive System** 
        based on the official installation manual (Rev. 7.28.15C).
        """)
        
        st.markdown("### 🎯 What You Can Ask")
        st.markdown("""
        - Installation procedures
        - Drive unit alignment
        - Tool requirements
        - Flooring installation
        - Hydraulic system setup
        - Troubleshooting guidance
        - Safety warnings
        - Technical specifications
        """)
        
        st.markdown("### ⚠️ Important Notice")
        st.warning("""
        Always refer to the complete official manual for safety-critical procedures. 
        This assistant is for guidance only.
        """)
        
        st.markdown("---")
        st.markdown("**KEITH Manufacturing Co.**")
        st.markdown("📞 800-547-6161")
        st.markdown("🌐 www.keithwalkingfloor.com")
    
    # Initialize RAG system
    rag = initialize_rag()
    
    if rag is None:
        st.error("System initialization failed. Please check your configuration.")
        return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Installation Question")
        
        # Example questions
        example_questions = [
            "What are the main steps for installing the drive unit?",
            "How do I align the drive unit properly?",
            "What tools are needed for installation?",
            "How do I install the floor seals?",
            "What is the recommended torque for floor bolts?",
            "How do I prepare the trailer for installation?"
        ]
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Example: How do I align the drive unit in a center frame trailer?",
            help="Ask anything about the Running Floor II installation process"
        )
        
        # Example questions dropdown
        selected_example = st.selectbox(
            "Or select an example question:",
            [""] + example_questions,
            help="Choose from common installation questions"
        )
        
        # Use selected example if provided
        if selected_example:
            question = selected_example
        
        # Submit button
        if st.button("🔍 Get Answer", use_container_width=True):
            if question.strip():
                with st.spinner("🤔 Analyzing installation manual..."):
                    result = rag.answer_question(question)
                    
                    # Display question
                    st.markdown(f'<div class="question-box"><strong>❓ Your Question:</strong><br/>{question}</div>', 
                               unsafe_allow_html=True)
                    
                    if result["success"]:
                        # Display answer
                        st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br/>{result["answer"]}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Display sources
                        st.markdown("### 📚 Referenced Manual Sections")
                        for i, source in enumerate(result["sources"], 1):
                            with st.expander(f"📄 Source {i}: {source['section']} (Page {source['page']})"):
                                st.markdown(f"**Section:** {source['section']}")
                                st.markdown(f"**Page:** {source['page']}")
                                st.markdown(f"**Preview:** {source['preview']}")
                    else:
                        st.error(f"❌ Error: {result['answer']}")
            else:
                st.warning("⚠️ Please enter a question or select an example.")
    
    with col2:
        st.markdown("### 📖 Manual Overview")
        
        # Manual statistics
        st.metric("Total Pages", "55")
        st.metric("Chapters", "7")
        st.metric("Appendices", "5")
        
        st.markdown("---")
        
        st.markdown("### 📑 Manual Sections")
        sections = {
            "Introduction": "1",
            "Trailer Prep": "2-8",
            "Sub-Deck": "9-16",
            "Drive Unit": "16-21",
            "Flooring": "22-35",
            "Hydraulic": "36-37",
            "Misc": "38-41",
            "Appendices": "42-55"
        }
        
        for section, pages in sections.items():
            st.markdown(f'<span class="keith-badge">{section}</span> Pages {pages}', 
                       unsafe_allow_html=True)
            st.markdown("")
        
        st.markdown("---")
        
        st.markdown("### ⏱️ Installation Time")
        st.info("**Estimated:** 35-100 hours\n\nDepends on experience and trailer adaptability")
        
        st.markdown("### 🔧 Key Systems")
        st.markdown("""
        - Drive Unit (Center Frame or Frameless)
        - Sub-Deck Structure
        - Floor Slats & Bearings
        - Hydraulic System
        - Side Seals
        - Front Shield
        """)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("**🏢 KEITH Manufacturing Co.**")
        st.markdown("World Headquarters")
    
    with col_f2:
        st.markdown("**📞 Contact**")
        st.markdown("800-547-6161")
        st.markdown("541-475-3802")
    
    with col_f3:
        st.markdown("**⚠️ Safety First**")
        st.markdown("Always follow official manual procedures")
    
    # Session state for chat history (optional future enhancement)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


if __name__ == "__main__":
    main()
