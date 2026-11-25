
import streamlit as st
import pandas as pd

from src.rag_pipeline import (
    build_research_index,
    answer_research_question
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(
            120deg,
            #172554,
            #312e81,
            #1e1b4b
        );
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 38px;
    }

    .hero p {
        color: #dbeafe;
        font-size: 17px;
    }

    .metric-card {
        background-color: #1c2333;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30394d;
    }

    .citation {
        background-color: #172033;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "research_index" not in st.session_state:
    st.session_state.research_index = None

if "uploaded_papers" not in st.session_state:
    st.session_state.uploaded_papers = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==========================================
# API KEY
# ==========================================

groq_api_key = st.secrets.get("GROQ_API_KEY", "")


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 ResearchMind AI")

    st.caption("Advanced AI Research Assistant")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Paper Library",
            "Ask AI",
            "RAG Analytics"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    if st.session_state.research_index:
        st.success("Research index ready")
    else:
        st.warning("No research index")

    if groq_api_key:
        st.success("Groq API configured")
    else:
        st.error("Groq API key missing")

    st.divider()

    st.caption("ResearchMind AI | RAG Research Project")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>ResearchMind AI</h1>
        <p>
        Your intelligent academic research companion.
        Upload research papers, ask questions and explore
        evidence-backed answers using advanced RAG.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Research Workspace")

    col1, col2, col3 = st.columns(3)

    paper_count = len(st.session_state.uploaded_papers)

    chunk_count = 0
    if st.session_state.research_index:
        chunk_count = len(
            st.session_state.research_index["chunks"]
        )

    with col1:
        st.metric("Uploaded Papers", paper_count)

    with col2:
        st.metric("Indexed Chunks", chunk_count)

    with col3:
        st.metric(
            "Questions Asked",
            len(st.session_state.chat_history)
        )

    st.divider()

    st.subheader("How ResearchMind AI Works")

    st.markdown("""
    1. **Upload:** Add multiple academic PDF documents.
    2. **Process:** Extract text and split it into chunks.
    3. **Index:** Generate embeddings and build search indexes.
    4. **Retrieve:** Combine semantic and keyword search.
    5. **Rerank:** Select the most relevant passages.
    6. **Answer:** Generate evidence-grounded responses
       with paper and page citations.
    """)

    st.info(
        "Start by opening Paper Library and uploading "
        "your research PDFs."
    )


# ==========================================
# PAPER LIBRARY
# ==========================================

elif page == "Paper Library":

    st.title("📂 Research Paper Library")

    st.write(
        "Upload multiple research papers in PDF format "
        "to build your searchable knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"Selected {len(uploaded_files)} research paper(s)."
        )

        if st.button(
            "Process Papers and Build AI Index",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Extracting PDFs, generating embeddings, "
                "and preparing search indexes..."
            ):

                try:
                    research_index = build_research_index(
                        uploaded_files
                    )

                    st.session_state.research_index = (
                        research_index
                    )

                    st.session_state.uploaded_papers = [
                        {
                            "name": file.name,
                            "size": file.size
                        }
                        for file in uploaded_files
                    ]

                    st.session_state.chat_history = []

                    st.success(
                        "Research papers successfully processed!"
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"Paper processing failed: {error}"
                    )

    st.divider()

    st.subheader("Indexed Paper Collection")

    if st.session_state.uploaded_papers:

        for paper in st.session_state.uploaded_papers:

            st.markdown(
                f"📄 **{paper['name']}** "
                f"— {paper['size'] / 1024:.1f} KB"
            )

        if st.session_state.research_index:

            st.success("These papers are indexed and searchable.")

    else:
        st.info("No papers have been indexed yet.")


# ==========================================
# ASK AI
# ==========================================

elif page == "Ask AI":

    st.title("💬 Ask ResearchMind AI")

    st.write(
        "Ask questions about your uploaded research papers. "
        "Answers are generated from retrieved paper passages."
    )

    if not st.session_state.research_index:

        st.warning(
            "Please upload and process research papers "
            "in the Paper Library first."
        )

    elif not groq_api_key:

        st.error(
            "Groq API key is missing. Configure "
            "GROQ_API_KEY in Streamlit secrets."
        )

    else:

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("citations"):
                    with st.expander("View source citations"):
                        for citation in message["citations"]:
                            st.markdown(
                                f"""
                                <div class="citation">
                                <b>Paper:</b> {citation['source']}<br>
                                <b>Page:</b> {citation['page']}<br>
                                <b>Reranking score:</b>
                                {citation['relevance_score']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        question = st.chat_input(
            "Ask a question about your research papers..."
        )

        if question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching papers and generating an answer..."
                ):

                    try:

                        result = answer_research_question(
                            question=question,
                            research_index=(
                                st.session_state.research_index
                            ),
                            groq_api_key=groq_api_key
                        )

                        answer = result["answer"]
                        citations = result["citations"]

                        st.markdown(answer)

                        if citations:

                            with st.expander(
                                "📖 View retrieved sources"
                            ):

                                for citation in citations:

                                    st.markdown(
                                        f"""
                                        <div class="citation">
                                        <b>Paper:</b>
                                        {citation['source']}<br>
                                        <b>Page:</b>
                                        {citation['page']}<br>
                                        <b>Reranking score:</b>
                                        {citation['relevance_score']}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "citations": citations
                            }
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to generate answer: {error}"
                        )


# ==========================================
# RAG ANALYTICS
# ==========================================

elif page == "RAG Analytics":

    st.title("📊 RAG Analytics")

    if not st.session_state.research_index:

        st.info(
            "Process research papers first to view analytics."
        )

    else:

        index = st.session_state.research_index

        chunk_stats = index["chunk_statistics"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Chunks",
                chunk_stats["total_chunks"]
            )

        with col2:
            st.metric(
                "Average Chunk Size",
                chunk_stats["average_chunk_size"]
            )

        with col3:
            st.metric(
                "Total Papers",
                len(index["papers"])
            )

        st.divider()

        st.subheader("Paper Processing Statistics")

        statistics_df = pd.DataFrame(
            index["statistics"]
        )

        st.dataframe(
            statistics_df,
            use_container_width=True
        )

        st.subheader("Indexed Research Papers")

        papers_df = pd.DataFrame(
            index["papers"]
        )

        st.dataframe(
            papers_df,
            use_container_width=True
        )

        st.caption(
            "Reranking scores are model ranking scores, "
            "not probabilities or confidence percentages."
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "ResearchMind AI | Hybrid Retrieval • RRF • "
    "Cross-Encoder Reranking • Grounded Generation"
)
