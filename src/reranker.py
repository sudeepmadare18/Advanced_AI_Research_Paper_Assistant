
from sentence_transformers import CrossEncoder


# -----------------------------------------
# RERANKER CONFIGURATION
# -----------------------------------------

DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -----------------------------------------
# LOAD CROSS-ENCODER
# -----------------------------------------

def create_reranker(
    model_name=DEFAULT_RERANKER_MODEL
):
    """
    Load a Cross-Encoder model for relevance scoring.

    Unlike bi-encoder embeddings, a Cross-Encoder
    evaluates the question and passage together.
    """

    try:

        model = CrossEncoder(
            model_name,
            max_length=512
        )

        return model

    except Exception as error:

        raise RuntimeError(
            f"Unable to load reranking model: {error}"
        )


# -----------------------------------------
# RERANK RETRIEVED DOCUMENTS
# -----------------------------------------

def rerank_documents(
    question,
    retrieved_results,
    reranker,
    top_n=5
):
    """
    Score and reorder retrieved passages according
    to their relevance to the user's question.

    Input:
        question: User's research question.
        retrieved_results: Results from hybrid search.
        reranker: Loaded CrossEncoder model.
        top_n: Number of final results to return.
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not retrieved_results:
        return []

    try:

        # Prepare question-passage pairs
        pairs = []

        for result in retrieved_results:

            document = result["document"]

            pairs.append([
                question,
                document.page_content
            ])

        # Predict relevance scores
        scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

        # Attach scores to results
        reranked_results = []

        for result, score in zip(
            retrieved_results,
            scores
        ):

            reranked_results.append({
                "document": result["document"],
                "rrf_score": result.get("rrf_score"),
                "rerank_score": float(score)
            })

        # Sort by Cross-Encoder relevance score
        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_n]

    except Exception as error:

        raise RuntimeError(
            f"Document reranking failed: {error}"
        )


# -----------------------------------------
# RERANKING SUMMARY
# -----------------------------------------

def get_reranking_summary(results):
    """
    Generate a simple summary of reranked passages.
    """

    summary = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        summary.append({
            "rank": rank,
            "source": document.metadata.get(
                "source", "Unknown"
            ),
            "page": document.metadata.get(
                "page_number", "Unknown"
            ),
            "relevance_score": round(
                result["rerank_score"], 4
            )
        })

    return summary
