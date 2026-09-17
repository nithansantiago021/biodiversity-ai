from pathlib import Path
from pypdf import PdfReader
from sqlalchemy.orm import Session
from app.models.db_models import Document, DocumentChunk


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page boundaries.
    Returns a list of dictionaries containing page number and page text.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue

        pages.append(
            {
                "page_number": page_number,
                "text": text,
            }
        )

    return pages


def create_document(
    db: Session,
    *,
    title: str,
    source: str,
    organization: str,
    publication_date=None,
    url: str | None = None,
    document_type: str = "report",
) -> Document:
    """
    Create and persist a Document record.
    """
    document = Document(
        title=title,
        source=source,
        organization=organization,
        publication_date=publication_date,
        url=url,
        document_type=document_type,
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return document


def create_document_chunks(
    db: Session,
    document: Document,
    pages: list[dict],
    *,
    chunk_size: int = 1000,
) -> list[DocumentChunk]:
    """
    Split extracted page text into chunks while preserving page provenance.
    A chunk never crosses a page boundary in this initial implementation.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        for start in range(0, len(text), chunk_size):
            chunk_text = text[start : start + chunk_size].strip()

            if not chunk_text:
                continue

            chunk = DocumentChunk(
                document_id=document.id,
                chunk_text=chunk_text,
                page_number=page_number,
                chunk_metadata={
                    "source_page": page_number,
                },
            )
            chunks.append(chunk)

    # Bulk add and commit efficiently
    db.add_all(chunks)
    db.commit()

    return chunks
