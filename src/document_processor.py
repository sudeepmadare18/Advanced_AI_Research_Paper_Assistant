
from io import BytesIO
from pypdf import PdfReader


def extract_pdf_metadata(pdf_file):
    """
    Extract basic metadata from an uploaded research paper.
    """

    try:
        pdf_bytes = pdf_file.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))

        metadata = reader.metadata

        return {
            "title": metadata.title if metadata and metadata.title else pdf_file.name,
            "author": metadata.author if metadata and metadata.author else "Unknown",
            "subject": metadata.subject if metadata and metadata.subject else "Not available",
            "total_pages": len(reader.pages),
            "filename": pdf_file.name
        }

    except Exception as error:
        raise ValueError(
            f"Unable to read PDF metadata: {error}"
        )


def extract_pdf_pages(pdf_file):
    """
    Extract text from every page of a research paper.

    Each page is stored separately so that our RAG system
    can provide page-level citations.
    """

    try:
        pdf_bytes = pdf_file.getvalue()
        reader = PdfReader(BytesIO(pdf_bytes))

        if reader.is_encrypted:
            raise ValueError(
                "Password-protected PDFs are not supported."
            )

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            cleaned_text = " ".join(text.split())

            if cleaned_text.strip():

                extracted_pages.append({
                    "text": cleaned_text,
                    "page_number": page_number,
                    "source": pdf_file.name,
                    "title": (
                        reader.metadata.title
                        if reader.metadata and reader.metadata.title
                        else pdf_file.name
                    )
                })

        if not extracted_pages:
            raise ValueError(
                "No readable text found. This may be a scanned PDF. "
                "OCR support will be added separately."
            )

        return extracted_pages

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            f"PDF text extraction failed: {error}"
        )


def process_research_paper(pdf_file):
    """
    Complete PDF processing pipeline.

    Returns:
        metadata: Information about the research paper.
        pages: Extracted page-wise text.
        statistics: Basic extraction statistics.
    """

    metadata = extract_pdf_metadata(pdf_file)

    pages = extract_pdf_pages(pdf_file)

    total_characters = sum(
        len(page["text"]) for page in pages
    )

    statistics = {
        "total_pages": metadata["total_pages"],
        "processed_pages": len(pages),
        "total_characters": total_characters,
        "average_characters_per_page": round(
            total_characters / len(pages), 2
        )
    }

    return {
        "metadata": metadata,
        "pages": pages,
        "statistics": statistics
    }
