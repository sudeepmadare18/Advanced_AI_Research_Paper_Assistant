
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever


# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

DEFAULT_TOP_K = 5
RRF_K = 60


# -----------------------------------------
# BUILD VECTOR DATABASE
# -----------------------------------------

def create_vector_database(
    documents,
    embedding_model
):
    """
    Store research paper chunks in ChromaDB.
    """

    if not documents:
        raise ValueError(
            "No documents available for vector storage."
        )

    try:

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name="research_papers"
        )

        return vectorstore

    except Exception as error:

        raise RuntimeError(
            f"ChromaDB creation failed: {error}"
        )


# -----------------------------------------
# BUILD BM25 RETRIEVER
# -----------------------------------------

def create_bm25_retriever(
    documents,
    top_k=DEFAULT_TOP_K
):
    """
    Create a keyword-based BM25 retriever.
    """

    if not documents:
        raise ValueError(
            "No documents available for BM25 retrieval."
        )

    try:

        retriever = BM25Retriever.from_documents(
            documents
        )

        retriever.k = top_k

        return retriever

    except Exception as error:

        raise RuntimeError(
            f"BM25 retriever creation failed: {error}"
        )


# -----------------------------------------
# RECIPROCAL RANK FUSION
# -----------------------------------------

def reciprocal_rank_fusion(
    result_lists,
    rrf_k=RRF_K
):
    """
    Combine ranked results from multiple retrievers.

    RRF gives higher importance to documents that
    appear near the top of multiple search result lists.
    """

    scores = {}
    document_map = {}

    for results in result_lists:

        for rank, document in enumerate(results, start=1):

            document_id = (
                document.metadata.get("chunk_id")
                or (
                    document.metadata.get("source", ""),
                    document.metadata.get("page_number", ""),
                    document.page_content
                )
            )

            if document_id not in scores:

                scores[document_id] = 0.0
                document_map[document_id] = document

            scores[document_id] += (
                1.0 / (rrf_k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=lambda item: scores[item],
        reverse=True
    )

    return [
        {
            "document": document_map[document_id],
            "rrf_score": scores[document_id]
        }
        for document_id in ranked_ids
    ]


# -----------------------------------------
# HYBRID SEARCH
# -----------------------------------------

def hybrid_search(
    question,
    vectorstore,
    bm25_retriever,
    top_k=DEFAULT_TOP_K
):
    """
    Perform vector search and BM25 keyword search,
    then combine the results using RRF.
    """

    if not question or not question.strip():

        raise ValueError(
            "Search question cannot be empty."
        )

    try:

        # Vector similarity search
        vector_results = vectorstore.similarity_search(
            question,
            k=top_k
        )

        # Keyword search
        bm25_results = bm25_retriever.invoke(
            question
        )

        bm25_results = bm25_results[:top_k]

        # Combine both ranked result lists
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results]
        )

        return fused_results[:top_k]

    except Exception as error:

        raise RuntimeError(
            f"Hybrid retrieval failed: {error}"
        )


# -----------------------------------------
# FORMAT RETRIEVED CONTEXT
# -----------------------------------------

def format_retrieved_context(
    retrieved_results
):
    """
    Convert retrieved documents into readable context
    for the LLM and preserve source/page information.
    """

    context_parts = []

    for index, result in enumerate(
        retrieved_results,
        start=1
    ):

        document = result["document"]

        source = document.metadata.get(
            "source", "Unknown source"
        )

        page = document.metadata.get(
            "page_number", "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {index}
Paper: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n\n".join(context_parts)
