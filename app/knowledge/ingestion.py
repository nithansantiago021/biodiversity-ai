import re
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


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuation boundaries."""
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def create_document_chunks(
    db: Session,
    document: Document,
    pages: list[dict],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Creates document chunks from page text, respecting natural sentence boundaries
    and maintaining a controlled overlap, while preserving page-level provenance.
    """
    chunks_to_add = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text:
            continue

        sentences = split_into_sentences(page_text)

        current_chunk_sentences = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for space

            # If adding this sentence exceeds chunk_size and we already have content, finalize chunk
            if current_length + sentence_len > chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks_to_add.append(
                    DocumentChunk(
                        document_id=document.id,
                        page_number=page_number,
                        chunk_text=chunk_text,
                        chunk_metadata={"source_page": page_number},
                    )
                )

                # Keep trailing sentence(s) for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_length + len(s) + 1 <= chunk_overlap or not overlap_sentences:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_length = overlap_length

            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        # Finalize remaining sentences on the page
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks_to_add.append(
                DocumentChunk(
                    document_id=document.id,
                    page_number=page_number,
                    chunk_text=chunk_text,
                    chunk_metadata={"source_page": page_number},
                )
            )

    db.add_all(chunks_to_add)
    db.commit()
    return chunks_to_add