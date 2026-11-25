
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------------
# CHUNKING CONFIGURATION
# -----------------------------------------

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


# -----------------------------------------
# CREATE TEXT SPLITTER
# -----------------------------------------

def create_text_splitter(
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """
    Create a recursive text splitter.

    Chunk overlap preserves some surrounding context
    between consecutive chunks.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative "
            "and smaller than chunk size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ],
        length_function=len,
        add_start_index=True
    )


# -----------------------------------------
# CONVERT PAGES INTO DOCUMENTS
# -----------------------------------------

def create_page_documents(pages):
    """
    Convert extracted page data into LangChain Documents.

    Every page retains its original source and page number.
    """

    documents = []

    for page in pages:

        text = page.get("text", "").strip()

        if not text:
            continue

        metadata = {
            "source": page.get("source", "Unknown"),
            "title": page.get("title", "Unknown"),
            "page_number": int(page.get("page_number", 0))
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return documents


# -----------------------------------------
# SPLIT DOCUMENTS INTO CHUNKS
# -----------------------------------------

def split_documents(
    pages,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """
    Split page documents into smaller searchable chunks.

    Each chunk retains its original page number and source.
    """

    if not pages:
        raise ValueError(
            "No extracted pages available for chunking."
        )

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    page_documents = create_page_documents(pages)

    chunks = splitter.split_documents(page_documents)

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['source']}_"
            f"page_{chunk.metadata['page_number']}_"
            f"chunk_{index}"
        )

        chunk.metadata["chunk_index"] = index

    if not chunks:
        raise ValueError(
            "No text chunks were generated."
        )

    return chunks


# -----------------------------------------
# CHUNKING STATISTICS
# -----------------------------------------

def get_chunk_statistics(chunks):
    """
    Return useful statistics about generated chunks.
    """

    if not chunks:
        return {
            "total_chunks": 0,
            "average_chunk_size": 0,
            "minimum_chunk_size": 0,
            "maximum_chunk_size": 0
        }

    sizes = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    return {
        "total_chunks": len(chunks),
        "average_chunk_size": round(
            sum(sizes) / len(sizes), 2
        ),
        "minimum_chunk_size": min(sizes),
        "maximum_chunk_size": max(sizes)
    }
