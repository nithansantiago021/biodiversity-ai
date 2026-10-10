import re
from pathlib import Path
from pypdf import PdfReader
from sqlalchemy.orm import Session
from typing import Any, Dict

from app.models.db_models import Document, DocumentChunk
from app.knowledge.embeddings import generate_chunk_embeddings


def extract_document_pages(file_path: str) -> list[dict]:
    """Extract text from PDF, TXT, or MD files while preserving page/section structure."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages: list[Dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_number": page_number, "text": text})
        return pages

    elif ext in [".txt", ".md"]:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        # Treat entire text/markdown file as page 1
        return [{"page_number": 1, "text": content}]

    else:
        raise ValueError(f"Unsupported file format: {ext}. Allowed: .pdf, .txt, .md")


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
    """Create and persist a Document record if it doesn't already exist."""
    existing_doc = db.query(Document).filter(Document.title == title).first()
    if existing_doc:
        return existing_doc

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
    """Creates document chunks respecting sentence boundaries and persists them to DB."""
    existing_chunks = (
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
    )
    if existing_chunks:
        return existing_chunks

    chunks_to_add: list[str] = []
    chunk_index = 0

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        if not page_text:
            continue

        sentences = split_into_sentences(page_text)
        current_chunk_sentences: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1

            if current_length + sentence_len > chunk_size and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunks_to_add.append(
                    DocumentChunk(
                        document_id=document.id,
                        page_number=page_number,
                        chunk_text=chunk_text,
                        chunk_metadata={
                            "source_page": page_number,
                            "chunk_index": chunk_index,
                            "document_title": document.title,
                            "organization": document.organization,
                            "document_type": document.document_type,
                        },
                    )
                )
                chunk_index += 1

                overlap_sentences: list[str] = []
                overlap_length = 0
                for s in reversed(current_chunk_sentences):
                    if (
                        overlap_length + len(s) + 1 <= chunk_overlap
                        or not overlap_sentences
                    ):
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_length = overlap_length

            current_chunk_sentences.append(sentence)
            current_length += sentence_len

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks_to_add.append(
                DocumentChunk(
                    document_id=document.id,
                    page_number=page_number,
                    chunk_text=chunk_text,
                    chunk_metadata={
                        "source_page": page_number,
                        "chunk_index": chunk_index,
                        "document_title": document.title,
                        "organization": document.organization,
                        "document_type": document.document_type,
                    },
                )
            )
            chunk_index += 1

    db.add_all(chunks_to_add)
    db.commit()

    return (
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
    )


def run_ingestion_pipeline(db: Session, raw_data_dir: str = "data/raw_data"):
    """Scans data/raw_data directory, ingests all files, and computes embeddings."""
    data_path = Path(raw_data_dir)
    if not data_path.exists():
        return

    supported_files = [
        f for f in data_path.iterdir() if f.suffix.lower() in [".pdf", ".txt", ".md"]
    ]

    for file_path in supported_files:
        pages = extract_document_pages(str(file_path))
        doc = create_document(
            db,
            title=file_path.stem.replace("_", " ").title(),
            source=file_path.name,
            organization="Environmental Research Institute",
            document_type="report" if file_path.suffix == ".pdf" else "notes",
        )
        chunks = create_document_chunks(db, doc, pages)
        generate_chunk_embeddings(db, chunks)


if __name__ == "__main__":
    from app.database import SessionLocal

    print("Starting document ingestion pipeline...")
    db = SessionLocal()
    try:
        run_ingestion_pipeline(db, raw_data_dir="data/raw_data")
        print("Ingestion pipeline completed successfully.")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        db.rollback()
    finally:
        db.close()
