
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------------------
# EMBEDDING MODEL CONFIGURATION
# -----------------------------------------

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embedding_model(
    model_name=DEFAULT_EMBEDDING_MODEL
):
    """
    Initialize a Hugging Face sentence embedding model.

    This model converts research paper text and user
    questions into numerical vectors.

    The same model must be used for both documents
    and queries to ensure compatible vector spaces.
    """

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,

            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )

        return embeddings

    except Exception as error:

        raise RuntimeError(
            f"Failed to initialize embedding model: {error}"
        )


# -----------------------------------------
# DOCUMENT EMBEDDING
# -----------------------------------------

def generate_document_embeddings(
    texts,
    embedding_model
):
    """
    Convert multiple research text chunks into vectors.
    """

    if not texts:
        raise ValueError(
            "No text chunks provided for embedding."
        )

    try:

        vectors = embedding_model.embed_documents(texts)

        return vectors

    except Exception as error:

        raise RuntimeError(
            f"Document embedding failed: {error}"
        )


# -----------------------------------------
# QUERY EMBEDDING
# -----------------------------------------

def generate_query_embedding(
    question,
    embedding_model
):
    """
    Convert a user's research question into a vector.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    try:

        vector = embedding_model.embed_query(
            question.strip()
        )

        return vector

    except Exception as error:

        raise RuntimeError(
            f"Query embedding failed: {error}"
        )
