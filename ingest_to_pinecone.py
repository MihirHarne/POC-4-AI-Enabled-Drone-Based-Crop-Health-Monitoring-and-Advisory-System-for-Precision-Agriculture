"""
=============================================================================
Dhan Saathi Knowledge Base - Pinecone Vector DB Ingestion Script
=============================================================================
This script:
1. Reads all 15 curated markdown files (11 diseases + 4 weedy rice guides).
2. Performs section-aware markdown chunking (keeping predisposing factors,
   dosages, and safety instructions intact).
3. Connects to Pinecone (via PINECONE_API_KEY).
4. Creates a Pinecone Serverless Index (default: 'dhan-saathi-vector-db').
5. Embeds chunks using Pinecone Integrated Inference (multilingual-e5-large)
   or local sentence-transformers, and upserts all vectors with rich metadata.
6. Runs a verification query to validate semantic search.
=============================================================================
"""

import os
import re
import glob
import time
from typing import List, Dict

# Optional: Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dhan-saathi-rice-kb")
EMBED_MODEL = "multilingual-e5-large"  # 1024 dimensions on Pinecone Inference
EMBED_DIMENSION = 1024


def parse_markdown_file(file_path: str) -> List[Dict]:
    """
    Parses a single markdown document into logical, section-aware text chunks.
    Preserves document title, category, pathogen, and section context.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract title from first # Header
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else os.path.basename(file_path)

    # Extract category if present
    category_match = re.search(r"\*\*Category\*\*:\s*([^\n]+)", content)
    category = category_match.group(1).strip() if category_match else "General Agronomy"

    # Extract causal agent if present
    causal_match = re.search(r"\*\*Causal Agent[^\*]*\*\*:\s*([^\n]+)", content)
    causal = causal_match.group(1).strip() if causal_match else ""

    # Split on major section headings (## and ###)
    sections = re.split(r"\n(?=#{2,3}\s+)", content)
    
    chunks = []
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue
            
        # Get section heading
        heading_match = re.match(r"^#{2,3}\s+(.+)$", sec, re.MULTILINE)
        section_name = heading_match.group(1).strip() if heading_match else ("Overview" if i == 0 else f"Section {i}")
        
        # Skip pure visual image galleries with no diagnostic text
        if "Visual Reference" in section_name or "Field Symptoms" in section_name:
            text_without_images = re.sub(r"!\[.*?\]\(.*?\)", "", sec).replace("-", "").replace("#", "").strip()
            if len(text_without_images) < 40:
                continue

        # Strip image markdown tags from text payload for dense text embeddings
        clean_sec = re.sub(r"!\[.*?\]\(.*?\)", "", sec).strip()
        
        # Require meaningful content (at least 50 alphanumeric characters)
        alnum_chars = len(re.findall(r"\w", clean_sec))
        if alnum_chars < 50:
            continue
            
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        sanitized_sec = re.sub(r'[^a-zA-Z0-9]+', '_', section_name).strip('_').lower()
        chunk_id = f"{doc_id}_{i}_{sanitized_sec}"[:64]
        
        header_context = f"Document: {title}"
        if causal:
            header_context += f" | Pathogen: {causal}"
        header_context += f" | Category: {category} | Section: {section_name}\n\n"
        
        chunk_payload = header_context + clean_sec
        
        chunks.append({
            "id": chunk_id,
            "title": title,
            "category": category,
            "causal_agent": causal,
            "section": section_name,
            "source": os.path.relpath(file_path, ".").replace("\\", "/"),
            "text": chunk_payload
        })
    return chunks


def load_all_markdown_chunks(base_dir: str = ".") -> List[Dict]:
    """Scans and parses all disease and weedy rice markdown files."""
    files = []
    files.extend(sorted(glob.glob(os.path.join(base_dir, "Knowledge_advisory", "disease_guides", "*.md"))))
    # files.extend(sorted(glob.glob(os.path.join(base_dir, "Knowledge_advisory", "weedy_rice_*.md"))))
    
    all_chunks = []
    for file_path in files:
        if os.path.basename(file_path) == "README.md":
            continue
        chunks = parse_markdown_file(file_path)
        all_chunks.extend(chunks)
        print(f"  ✓ Processed: {os.path.basename(file_path):45} -> {len(chunks)} chunks")
        
    return all_chunks


def ingest_to_pinecone(chunks: List[Dict]):
    """Connects to Pinecone, provisions the index, embeds chunks, and upserts."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        api_key = input("Enter your Pinecone API Key: ").strip()
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required to connect to Pinecone.")

    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        raise ImportError(
            "Pinecone client library is not installed.\n"
            "Please install it using: pip install pinecone"
        )

    print(f"\n[1/4] Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)

    # 1. Create Serverless Index if it doesn't already exist
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"[2/4] Index '{INDEX_NAME}' not found. Creating Pinecone Serverless Index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"      Waiting for index '{INDEX_NAME}' to initialize...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)
        print(f"  ✓ Index '{INDEX_NAME}' created and ready.")
    else:
        print(f"[2/4] Found existing index '{INDEX_NAME}'. Reusing it.")

    index = pc.Index(INDEX_NAME)

    # 2. Embed and Upsert in Batches using Pinecone Inference
    print(f"[3/4] Generating embeddings and upserting {len(chunks)} chunks...")
    batch_size = 32
    total_upserted = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        # Use Pinecone Inference API to embed chunks
        # model="multilingual-e5-large" creates 1024-dim dense vectors
        embeddings = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )

        vectors_to_upsert = []
        for j, record in enumerate(batch):
            vector_values = embeddings[j]["values"]
            metadata = {
                "text": record["text"],
                "title": record["title"],
                "category": record["category"],
                "causal_agent": record["causal_agent"],
                "section": record["section"],
                "source": record["source"]
            }
            vectors_to_upsert.append({
                "id": record["id"],
                "values": vector_values,
                "metadata": metadata
            })

        index.upsert(vectors=vectors_to_upsert)
        total_upserted += len(vectors_to_upsert)
        print(f"      Upserted {total_upserted}/{len(chunks)} chunks...")

    print(f"  ✓ Successfully stored {total_upserted} vector embeddings in Pinecone index '{INDEX_NAME}'.")

    # 3. Verification Query
    print(f"\n[4/4] Running semantic search test...")
    sample_query = "What fungicide and water quantity should I spray for brown spot disease?"
    print(f"      Test Query: '{sample_query}'")

    query_embedding = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[sample_query],
        parameters={"input_type": "query"}
    )[0]["values"]

    results = index.query(
        vector=query_embedding,
        top_k=2,
        include_metadata=True
    )

    print("\n--- Top Retrieved Results from Vector DB ---")
    for match in results["matches"]:
        score = match["score"]
        meta = match["metadata"]
        print(f"\n[Score: {score:.4f}] {meta['title']} ({meta['section']})")
        print(f"Source: {meta['source']}")
        print(f"Snippet:\n{meta['text'][:250]}...")


if __name__ == "__main__":
    print("=================================================================")
    print(" Dhan Saathi: Markdown Knowledge Base -> Pinecone Ingestion")
    print("=================================================================")
    chunks = load_all_markdown_chunks()
    print(f"\nTotal high-quality chunks parsed: {len(chunks)}")
    
    # Prompt before upserting
    proceed = input("\nDo you want to connect to Pinecone and upsert these chunks now? (y/n): ").strip().lower()
    if proceed == 'y':
        ingest_to_pinecone(chunks)
    else:
        print("\nIngestion cancelled. You can run this script anytime once your PINECONE_API_KEY is ready:")
        print("  python ingest_to_pinecone.py")
