
import streamlit as st
from datetime import datetime

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# CUSTOM DESIGN
# -----------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

[data-testid="stSidebar"] * {
    color: #F9FAFB !important;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #172554;
}

.subtitle {
    font-size: 16px;
    color: #64748B;
}

.hero {
    background: linear-gradient(120deg, #172554, #164E63);
    padding: 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
}

.hero p {
    color: #DCEAFE;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    min-height: 115px;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #172554;
}

.section-heading {
    color: #172554;
    font-size: 21px;
    font-weight: 700;
}

.feature-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    min-height: 150px;
}

.feature-card h3 {
    color: #172554;
}

.feature-card p {
    color: #64748B;
    font-size: 14px;
}

div.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# SESSION STATE
# -----------------------------------------

if "uploaded_papers" not in st.session_state:
    st.session_state.uploaded_papers = {}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# -----------------------------------------
# SIDEBAR
# -----------------------------------------

with st.sidebar:

    st.markdown("# 📚 ResearchMind")
    st.caption("ADVANCED AI RESEARCH WORKSPACE")

    st.divider()

    st.markdown("### WORKSPACE")

    pages = [
        "Dashboard",
        "Paper Library",
        "Ask AI",
        "Compare Papers",
        "Research Summary",
        "RAG Analytics"
    ]

    for page in pages:
        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True
        ):
            st.session_state.page = page

    st.divider()

    st.markdown("### SYSTEM STATUS")
    st.success("Application interface ready")

    st.caption("Powered by Groq + Hugging Face")


# -----------------------------------------
# COMMON DATA
# -----------------------------------------

paper_count = len(st.session_state.uploaded_papers)

# -----------------------------------------
# DASHBOARD
# -----------------------------------------

if st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Welcome to ResearchMind AI 👋</h1>
        <p>
        Your intelligent research companion.
        Upload academic papers, explore scientific
        knowledge and discover insights using advanced RAG.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Research Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Uploaded Papers</div>
            <div class="metric-value">{paper_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Knowledge Chunks</div>
            <div class="metric-value">--</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">RAG System</div>
            <div class="metric-value">Setup</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("## Explore Your Workspace")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📄 Paper Library</h3>
            <p>Upload and organize multiple academic research papers.</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Compare Papers</h3>
            <p>Compare research methods, objectives and findings.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Ask AI</h3>
            <p>Ask questions and receive answers supported by research sources.</p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <div class="feature-card">
            <h3>📊 RAG Analytics</h3>
            <p>Evaluate retrieval quality and answer performance.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.info(
        "Getting started: Open Paper Library and upload "
        "your first research paper."
    )


# -----------------------------------------
# PAPER LIBRARY
# -----------------------------------------

elif st.session_state.page == "Paper Library":

    st.title("📄 Research Paper Library")

    st.write(
        "Upload and manage your academic research documents."
    )

    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more academic PDF papers."
    )

    if uploaded_files:

        for file in uploaded_files:

            if file.name not in st.session_state.uploaded_papers:

                st.session_state.uploaded_papers[file.name] = {
                    "name": file.name,
                    "data": file.getvalue(),
                    "uploaded_at": datetime.now().strftime(
                        "%d %b %Y, %I:%M %p"
                    )
                }

        st.success("Research papers added to your library.")

    st.divider()

    st.subheader("Your Documents")

    if st.session_state.uploaded_papers:

        for name, paper in st.session_state.uploaded_papers.items():

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"**📕 {name}**")
                    st.caption(
                        f"Uploaded: {paper['uploaded_at']}"
                    )

                with col2:
                    if st.button(
                        "Remove",
                        key=f"remove_{name}"
                    ):
                        del st.session_state.uploaded_papers[name]
                        st.rerun()

    else:
        st.info("No research papers uploaded yet.")


# -----------------------------------------
# ASK AI
# -----------------------------------------

elif st.session_state.page == "Ask AI":

    st.title("💬 Ask ResearchMind AI")

    st.write(
        "Ask questions about your uploaded research papers."
    )

    if not paper_count:
        st.warning(
            "Please upload research papers in Paper Library first."
        )

    question = st.text_area(
        "Your research question",
        placeholder=(
            "Example: What are the main limitations "
            "of the proposed methodology?"
        ),
        height=120
    )

    if st.button("Generate Answer", type="primary"):

        if not paper_count:
            st.warning("Upload research papers first.")

        elif not question.strip():
            st.warning("Enter a question.")

        else:
            st.info(
                "Question received. The RAG retrieval and "
                "Groq generation pipeline will be connected "
                "in the next development stage."
            )


# -----------------------------------------
# COMPARE PAPERS
# -----------------------------------------

elif st.session_state.page == "Compare Papers":

    st.title("🔍 Research Paper Comparison")

    st.write(
        "Compare objectives, methodologies, findings "
        "and limitations across academic papers."
    )

    available_papers = list(
        st.session_state.uploaded_papers.keys()
    )

    if len(available_papers) < 2:

        st.warning(
            "Upload at least two research papers to use comparison."
        )

    else:

        selected = st.multiselect(
            "Select papers to compare",
            available_papers,
            max_selections=5
        )

        if st.button("Compare Selected Papers"):

            if len(selected) < 2:
                st.warning("Select at least two papers.")

            else:
                st.info(
                    "AI-based comparison will be connected "
                    "after the RAG pipeline is implemented."
                )


# -----------------------------------------
# RESEARCH SUMMARY
# -----------------------------------------

elif st.session_state.page == "Research Summary":

    st.title("📝 Research Paper Summary")

    st.write(
        "Generate structured summaries of your academic documents."
    )

    available_papers = list(
        st.session_state.uploaded_papers.keys()
    )

    if available_papers:

        selected_paper = st.selectbox(
            "Choose a research paper",
            available_papers
        )

        if st.button("Generate Summary"):

            st.info(
                "Summarization will be connected to the "
                "document processing and Groq pipeline."
            )

    else:
        st.warning("Upload a paper before generating summaries.")


# -----------------------------------------
# RAG ANALYTICS
# -----------------------------------------

elif st.session_state.page == "RAG Analytics":

    st.title("📊 RAG Performance Analytics")

    st.write(
        "Monitor retrieval performance and evaluate "
        "AI-generated research answers."
    )

    st.info(
        "Analytics will become available after we implement "
        "document chunking, retrieval, reranking and evaluation."
    )

    st.markdown("### Planned Evaluation Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Retrieval Metrics**")
        st.write("- Context Precision")
        st.write("- Context Recall")
        st.write("- Retrieval Relevance")

    with col2:
        st.markdown("**Answer Metrics**")
        st.write("- Faithfulness")
        st.write("- Answer Relevance")
        st.write("- Citation Accuracy")


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "ResearchMind AI | Advanced Retrieval-Augmented Generation "
    "(RAG) Research Assistant"
)
