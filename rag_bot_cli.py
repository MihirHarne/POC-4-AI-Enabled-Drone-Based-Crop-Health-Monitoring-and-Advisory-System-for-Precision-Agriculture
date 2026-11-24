"""
=============================================================================
Dhan Saathi - Agricultural RAG Advisory Bot (CLI)
=============================================================================
Powered by:
- Knowledge Base: 15 ICAR-NRRI & DWR Rice Disease & Weedy Rice Guides
- Vector Database: Pinecone (Serverless) + Pinecone Inference Embeddings
- LLM: Groq (Llama-3.3-70b-versatile OR gpt-oss-20b / Meta Llama 3)
=============================================================================
"""

import os
import sys
import time

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Pinecone & Groq SDKs
try:
    from pinecone import Pinecone
except ImportError:
    print("❌ Error: 'pinecone' package is not installed.")
    print("Please run: pip install pinecone")
    sys.exit(1)

try:
    from groq import Groq
except ImportError:
    print("❌ Error: 'groq' package is not installed.")
    print("Please run: pip install groq")
    sys.exit(1)


# Configuration Defaults
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dhan-saathi-rice-kb")
EMBED_MODEL = "multilingual-e5-large"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = 3


SYSTEM_PROMPT = """You are "Dhan Saathi", an expert agricultural advisor and precision rice crop specialist.
Your knowledge is strictly grounded in the official diagnostic guides from the ICAR - National Rice Research Institute (NRRI) and Directorate of Weed Research (DWR).

Instructions for your response:
1. Deliver practical, clear, actionable advice tailored for a rice farmer or agronomist.
2. Ground all chemical controls, product names (e.g., Tricyclazole, Propiconazole, Validamycin, Mancozeb), formulations, exact dosages (e.g., g/L, ml/L, or kg/ha), spray timings (e.g., boot leaf stage), and water volumes (200 L/acre for manual knapsack, 80 L/acre for automatic/power sprayers) STRICTLY on the retrieved context below.
3. If predisposing weather conditions (temperature, humidity, dew, rainfall) or cultural practices (seedling density, water level, straw burning, stale seedbed) are relevant to the query, explain them clearly.
4. If the retrieved context does not provide a safe or definitive answer, clearly say so and advise contacting the local Krishi Vigyan Kendra (KVK) or the national Kisan Call Centre (1800-180-1551). Never hallucinate or invent unauthorized chemicals or dosages.
5. Keep your tone respectful, supportive, and professional. Use formatting (bullet points, bold text) to make dosages and steps immediately readable.
"""


def get_api_keys():
    """Retrieves API keys from environment or user prompt."""
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        pinecone_key = input("Enter your PINECONE_API_KEY: ").strip()
        if not pinecone_key:
            print("❌ Pinecone API Key is required.")
            sys.exit(1)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        groq_key = input("Enter your GROQ_API_KEY: ").strip()
        if not groq_key:
            print("❌ Groq API Key is required.")
            sys.exit(1)

    return pinecone_key, groq_key


def initialize_clients(pinecone_key: str, groq_key: str):
    """Initializes Pinecone Index and Groq Client."""
    print("⏳ Connecting to Pinecone vector database...")
    pc = Pinecone(api_key=pinecone_key)
    
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"❌ Error: Index '{PINECONE_INDEX_NAME}' was not found in your Pinecone project.")
        print(f"Available indexes: {existing_indexes}")
        print("Please run 'python ingest_to_pinecone.py' first to build and populate your vector DB.")
        sys.exit(1)

    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✓ Connected to Pinecone index: '{PINECONE_INDEX_NAME}'")

    print(f"⏳ Initializing Groq client (Model: {GROQ_MODEL})...")
    groq_client = Groq(api_key=groq_key)
    print("✓ Groq client initialized.")

    return pc, index, groq_client


def retrieve_context(pc: Pinecone, index, query: str, top_k: int = TOP_K):
    """Embeds user query and searches Pinecone for the most relevant chunks."""
    # Generate query embedding using Pinecone Inference
    embed_response = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[query],
        parameters={"input_type": "query"}
    )
    query_vector = embed_response[0]["values"]

    # Search top-k matches
    search_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = search_results.get("matches", [])
    if not matches:
        return "", []

    context_blocks = []
    sources = []

    for i, match in enumerate(matches, 1):
        score = match.get("score", 0.0)
        meta = match.get("metadata", {})
        title = meta.get("title", "Unknown")
        section = meta.get("section", "General")
        source = meta.get("source", "")
        text = meta.get("text", "")

        context_blocks.append(
            f"--- Context Block [{i}] (Relevance Score: {score:.3f}) ---\n"
            f"{text}\n"
        )
        sources.append({
            "title": title,
            "section": section,
            "source": source,
            "score": score
        })

    full_context = "\n".join(context_blocks)
    return full_context, sources


def ask_rag_bot(groq_client: Groq, query: str, context: str):
    """Sends retrieved context and query to Groq LLM with streaming output."""
    user_message = f"""Retrieved Knowledge Context:
==================================================
{context}
==================================================

Farmer's Question: {query}

Please provide your grounded advisory response:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    response_stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,      # Low temperature for strict factual adherence
        max_tokens=1024,
        stream=True
    )

    print("\n🌾 Dhan Saathi:")
    print("----------------------------------------------------------------------")
    collected_text = []
    for chunk in response_stream:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
        collected_text.append(content)
    print("\n----------------------------------------------------------------------")
    return "".join(collected_text)


def print_banner():
    """Prints the CLI start banner."""
    print("\n" + "=" * 70)
    print("   🌾  DHAN SAATHI - PRECISION RICE ADVISORY RAG BOT (CLI)  🌾")
    print("=" * 70)
    print(f" • Vector Store : Pinecone Serverless ('{PINECONE_INDEX_NAME}')")
    print(f" • Embedding    : {EMBED_MODEL} (1024-dim)")
    print(f" • LLM Engine   : Groq / {GROQ_MODEL}")
    print(f" • Knowledge    : 11 ICAR Rice Diseases + 4 Weedy Rice Monographs")
    print("=" * 70)
    print("Commands:")
    print(" - Type your agricultural question in English and press Enter.")
    print(" - Type 'sources' to toggle viewing retrieved context blocks.")
    print(" - Type 'clear' to clear the terminal screen.")
    print(" - Type 'exit' or 'quit' to close the session.")
    print("=" * 70 + "\n")


def main():
    pinecone_key, groq_key = get_api_keys()
    pc, index, groq_client = initialize_clients(pinecone_key, groq_key)
    
    print_banner()
    show_sources = True

    while True:
        try:
            query = input("\n🧑‍🌾 Farmer > ").strip()
            
            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Thank you for using Dhan Saathi. May your fields prosper!\n")
                break

            if query.lower() == "clear":
                os.system("cls" if os.name == "nt" else "clear")
                print_banner()
                continue

            if query.lower() == "sources":
                show_sources = not show_sources
                status = "ON" if show_sources else "OFF"
                print(f"ℹ️  Retrieved source citations toggled: {status}")
                continue

            print("🔎 Searching ICAR knowledge base in Pinecone...", end="", flush=True)
            start_time = time.time()
            context, sources = retrieve_context(pc, index, query, top_k=TOP_K)
            search_duration = time.time() - start_time
            print(f" (Found {len(sources)} context blocks in {search_duration:.2f}s)")

            if not context:
                print("\n⚠️ No relevant advisory content found in the knowledge base.")
                print("Please try rephrasing your question or consult your local KVK officer.")
                continue

            # Stream LLM response
            ask_rag_bot(groq_client, query, context)

            # Display source citations
            if show_sources and sources:
                print("📚 Sources Consulted:")
                for s in sources:
                    print(f"   • {s['title']} [{s['section']}] (Score: {s['score']:.3f}) -> {s['source']}")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")


if __name__ == "__main__":
    main()
