
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.document_processor import process_research_paper
from src.chunking import split_documents, get_chunk_statistics
from src.embeddings import create_embedding_model
from src.retrieval import (
    create_vector_database,
    create_bm25_retriever,
    hybrid_search
)
from src.reranker import (
    create_reranker,
    rerank_documents
)


# -----------------------------------------
# MODEL CONFIGURATION
# -----------------------------------------

LLM_MODEL = "llama-3.3-70b-versatile"

RETRIEVAL_TOP_K = 10
FINAL_TOP_N = 5


# -----------------------------------------
# BUILD RESEARCH KNOWLEDGE BASE
# -----------------------------------------

def build_research_index(uploaded_files):
    """
    Process uploaded research papers and create
    the vector and keyword search indexes.
    """

    if not uploaded_files:
        raise ValueError("Upload at least one PDF.")

    embedding_model = create_embedding_model()

    all_chunks = []
    paper_metadata = []
    paper_statistics = []

    for pdf_file in uploaded_files:

        # Step 1: Extract PDF text
        processed = process_research_paper(pdf_file)

        # Step 2: Split extracted pages into chunks
        chunks = split_documents(
            processed["pages"]
        )

        all_chunks.extend(chunks)

        paper_metadata.append(
            processed["metadata"]
        )

        paper_statistics.append(
            processed["statistics"]
        )

    if not all_chunks:
        raise ValueError(
            "No searchable text was extracted from the PDFs."
        )

    # Step 3: Create ChromaDB vector index
    vectorstore = create_vector_database(
        all_chunks,
        embedding_model
    )

    # Step 4: Create BM25 keyword index
    bm25_retriever = create_bm25_retriever(
        all_chunks,
        top_k=RETRIEVAL_TOP_K
    )

    # Step 5: Load Cross-Encoder reranker
    reranker = create_reranker()

    return {
        "embedding_model": embedding_model,
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "reranker": reranker,
        "chunks": all_chunks,
        "papers": paper_metadata,
        "statistics": paper_statistics,
        "chunk_statistics": get_chunk_statistics(all_chunks)
    }


# -----------------------------------------
# RETRIEVE RELEVANT RESEARCH PASSAGES
# -----------------------------------------

def retrieve_research_context(
    question,
    research_index
):
    """
    Retrieve relevant passages using hybrid search
    and rerank them using a Cross-Encoder.
    """

    # Step 1: Hybrid retrieval
    retrieved_results = hybrid_search(
        question=question,
        vectorstore=research_index["vectorstore"],
        bm25_retriever=research_index["bm25_retriever"],
        top_k=RETRIEVAL_TOP_K
    )

    # Step 2: Cross-Encoder reranking
    final_results = rerank_documents(
        question=question,
        retrieved_results=retrieved_results,
        reranker=research_index["reranker"],
        top_n=FINAL_TOP_N
    )

    return final_results


# -----------------------------------------
# FORMAT SOURCE CITATIONS
# -----------------------------------------

def format_source_citations(results):
    """
    Prepare source and page references for the answer.
    """

    citations = []

    for result in results:

        document = result["document"]

        source = document.metadata.get(
            "source", "Unknown paper"
        )

        page = document.metadata.get(
            "page_number", "Unknown"
        )

        citation = {
            "source": source,
            "page": page,
            "relevance_score": round(
                result["rerank_score"], 4
            ),
            "content": document.page_content
        }

        citations.append(citation)

    return citations


# -----------------------------------------
# GROQ ANSWER GENERATION
# -----------------------------------------

def generate_research_answer(
    question,
    retrieved_results,
    groq_api_key
):
    """
    Generate an evidence-grounded answer using Groq.

    The LLM is instructed to use only the retrieved
    research passages and include citations.
    """

    if not groq_api_key:
        raise ValueError("Groq API key is missing.")

    if not retrieved_results:
        return (
            "I could not find relevant information in "
            "the uploaded research papers.",
            []
        )

    context_parts = []

    for index, result in enumerate(
        retrieved_results,
        start=1
    ):

        document = result["document"]

        source = document.metadata.get(
            "source", "Unknown paper"
        )

        page = document.metadata.get(
            "page_number", "Unknown"
        )

        context_parts.append(
            f"""
[Source {index}]
Paper: {source}
Page: {page}

Research passage:
{document.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are ResearchMind AI, an academic research assistant.

Your task is to answer questions using the supplied
research-paper passages.

STRICT RULES:

1. Use only information supported by the passages.
2. Never invent research findings, statistics or citations.
3. If the passages do not contain the answer, clearly
   say that the available papers do not provide enough
   information.
4. Explain technical concepts in clear language.
5. Compare findings only when evidence is available.
6. Add citations in this format:
   [Paper: filename, Page: number]
7. Distinguish reported findings from your interpretation.
8. Do not claim that you have read parts of a paper
   that were not supplied in the context.

USER QUESTION:
{question}

RETRIEVED RESEARCH PASSAGES:
{context}

Write a clear, structured answer with relevant citations.
"""

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        api_key=groq_api_key
    )

    response = llm.invoke(prompt)

    citations = format_source_citations(
        retrieved_results
    )

    return response.content, citations


# -----------------------------------------
# COMPLETE QUESTION-ANSWER PIPELINE
# -----------------------------------------

def answer_research_question(
    question,
    research_index,
    groq_api_key
):
    """
    Execute the complete RAG question-answer workflow.
    """

    if not question or not question.strip():
        raise ValueError("Enter a research question.")

    # Retrieve relevant passages
    retrieved_results = retrieve_research_context(
        question,
        research_index
    )

    # Generate grounded answer
    answer, citations = generate_research_answer(
        question=question,
        retrieved_results=retrieved_results,
        groq_api_key=groq_api_key
    )

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_count": len(retrieved_results)
    }
