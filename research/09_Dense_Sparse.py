"""
================================================================================
This script introduces the robust industry architecture known as 
**Hybrid RAG** or **Ensemble Retrieval**.
By blending semantic understanding with structural syntax matching, 
this design addresses the retrieval blind spots found in isolated 
dense or sparse strategies.

THE COMPONENT TRIAD:
- **Component A: Dense Retriever (The Smart Assistant)**: 
    Employs a local HuggingFace embedding model (`all-MiniLM-L6-v2`) and FAISS to decode 
    =conceptual meaning and synonyms=. Excellent for abstract, intent-driven searches.
- **Component B: Sparse Retriever (The Keyword Filter)**: 
    Leverages a traditional BM25 algorithm to look for =exact keyword= overlapping. Crucial for capturing 
    hyper-specific technical acronyms, names, or short numeric code strings.
- **Component C: Ensemble Layer (The Hybrid Orchestrator)**: 
    Fuses both retrievers together using a custom weighting spectrum (e.g., 70% Dense trust vs. 30% Sparse trust).

THE LOGICAL FLOW & DIAGNOSTIC TESTS:
1. INITIALIZATION: Builds separate Dense (FAISS) and Sparse (BM25) retriever channels.
2. HYBRID COMBINATION: Passes both streams into `EnsembleRetriever` to merge results.
3. RAG CHAIN CREATION: Assembles standard LangChain retrieval tools (`create_retrieval_chain`).
4. COMPARATIVE PROOF CASES: Runs target queries demonstrating where Dense triumphs 
   (deciphering abstract terms like "marine organism system" to locate a reef) 
   and where Sparse rules (catching exact hits for specific terms like "qubits").
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains.retrieval import create_retrieval_chain

# https://reference.langchain.com/python/langchain-huggingface/embeddings/huggingface/HuggingFaceEmbeddings
# https://reference.langchain.com/python/langchain-community/retrievers/bm25/BM25Retriever
# https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever

# https://reference.langchain.com/python/langchain-classic/chains/combine_documents/stuff/create_stuff_documents_chain
# https://reference.langchain.com/python/langchain-classic/chains/retrieval/create_retrieval_chain

# ==============================================================================
# 1. INITIAL PREPARATION
# ==============================================================================

# Silence logs and load environment
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
load_dotenv()

# Step 1: Define Sample Documents
docs = [
    Document(page_content="The Great Barrier Reef is the world's largest coral reef system."),
    Document(page_content="Warming ocean temperatures are causing severe bleaching across coral habitats."),
    Document(page_content="Quantum computing utilizes qubits to perform complex calculations at unprecedented speeds."),
    Document(page_content="Tech companies are heavily investing in quantum research to crack advanced cryptography."),
    Document(page_content="Sourdough bread relies on a wild yeast starter for its rise and distinct flavor."),
    Document(page_content="Baking the perfect loaf requires a long fermentation process and high oven heat.")
]

# ==============================================================================
# 2. RETRIEVER SETUP (HYBRID STRATEGY)
# ==============================================================================

# --- Component A: Dense Retriever (Semantic Similarity) Like a The Smart Assistant---
# Uses embeddings to understand the "meaning" of the query
# Employs a local HuggingFace embedding model (`all-MiniLM-L6-v2`) and 
# FAISS to decode =conceptual meaning and synonyms=. Excellent for abstract, intent-driven searches.
MODEL_NAME_EMBEDDING = "all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME_EMBEDDING)
dense_vectorstore = FAISS.from_documents(docs, embedding_model)
dense_retriever = dense_vectorstore.as_retriever()
dense_retriever.search_kwargs = {"k": 2}

# --- Component B: Sparse Retriever (Keyword Matching) Like Ctrl + F---
# If a query asks for "tips on sourdough fermentation," the sparse retriever might 
# completely miss a document that says "Baking the perfect loaf requires a long fermentation process,"
# because the exact word "sourdough" isn't present—even though the meaning is identical.
# Uses BM25 algorithm to find exact word matches

# Conversely, if a query asks for "qubits," the dense retriever might dilute its importance by 
# mixing it up with generic "computing" or "math" concepts. The sparse retriever is crucial here: 
# it uses the BM25 algorithm to guarantee that documents containing the exact technical keyword "qubits" 
# are forced to the top of the results.

sparse_retriever = BM25Retriever.from_documents(docs)
# sparse_retriever.k = 3
sparse_retriever.k = 1  # Force it to only grab the exact keyword hit


# --- Component C: Ensemble Retriever (The Hybrid Layer) ---
# Combines results from both. Weights determine which retriever to "trust" more.
# Here, 70% weight is given to Dense and 30% to Sparse.
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)

# ==============================================================================
# 3. RAG CHAIN CONSTRUCTION
# ==============================================================================

# Initialize LLM and Prompt
MODEL_NAME_LLM = "openai:gpt-4o-mini"
llm = init_chat_model(MODEL_NAME_LLM, temperature=0.2)
prompt = PromptTemplate.from_template("""
Answer the question based on the context below.
Context: {context}
Question: {input}
""")

### Create a document chain
# create_stuff_documents_chain creates a chain that "stuffs" (inserts) all retrieved documents 
# into a single prompt and sends it to the LLM. 
# It's called "stuff" because it literally stuffs all the documents into the context window at once.

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
### Create The Final RAG Chain
# create_retrieval_chain is a function that combines a retriever (which fetches relevant documents) 
# with a document chain (which processes those documents with an LLM) to create a complete RAG pipeline.
rag_chain = create_retrieval_chain(
    retriever=hybrid_retriever, 
    combine_docs_chain=document_chain
)

# ==============================================================================
# 4. EXECUTION
# ==============================================================================

query = {"input": "Tell me about the Great Barrier Reef and its conservation?"}
response = rag_chain.invoke(query)
print(f"🔍 Query: {query['input']}")
print(f"✅ Answer:\n{response['answer']}")

print("\n📄 Source Documents (Combined from Dense & Sparse):")
for i, doc in enumerate(response["context"]):
    print(f"Doc {i+1}: {doc.page_content}")


#================================================================================
# ==============================================================================
# 4. EXECUTION & HYBRID BENEFIT DEMONSTRATION
# ==============================================================================

# Helper function to print clean results
def run_hybrid_test(test_title, user_query):
    print(f"\n========================================================")
    print(f"🔥 {test_title}")
    print(f"========================================================")
    print(f"Query: '{user_query}'\n")
    
    response = rag_chain.invoke({"input": user_query})
    
    print(f"✅ LLM Answer:\n{response['answer']}\n")
    print("📄 Retrieved Source Chunks passed to LLM:")
    for i, doc in enumerate(response["context"]):
        print(f"  [{i+1}] {doc.page_content}")

# ------------------------------------------------------------------------------
# TEST 1: The Dense Advantage (Semantic/Synonym Match)
# ------------------------------------------------------------------------------
# Notice we use the word "organism" and "fading". The document does NOT contain 
# these exact words, but the Dense retriever understands the concept.
run_hybrid_test(
    "TEST 1: Dense Advantage (Synonyms & Conceptual Meaning)",
    "Tell me about the largest marine organism system and its fading colors."
)


# ------------------------------------------------------------------------------
# TEST 2: The Sparse Advantage (Exact Keyword Match)
# ------------------------------------------------------------------------------
# In a short document pool, an embedding model can easily mistake "qubits" as 
# being close to generic tech words. BM25 forces an exact match.
run_hybrid_test(
    "TEST 2: Sparse Advantage (Technical Keywords & Code Names)",
    "What tech companies are cracking cryptography using qubits?"
)