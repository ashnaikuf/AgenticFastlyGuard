# Re-ranking is a second-stage filtering process in retrieval systems, especially in RAG pipelines, where we:
# 1. First use a fast retriever (like BM25, FAISS, hybrid) to fetch top-k documents quickly.
# 2. Then use a more accurate but slower model (like a cross-encoder or LLM) to re-score and reorder those documents by relevance to the query.
# 👉 It ensures that the most relevant documents appear at the top, improving the final answer from the LLM.

import os
import logging
import warnings
from dotenv import load_dotenv

# LangChain Core & Text Splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 

# Models & Vector Stores
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Logging
# ==============================================================================

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Suppress noise: warnings, transformer loading logs, and progress bars
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ==============================================================================
# 2. DATA INGESTION & PROCESSING
# ==============================================================================

FILE_NAME = "cybersecurity_data/cybersecurity_concepts.txt"
# Load the local knowledge base
try:
    loader = TextLoader(FILE_NAME)
    raw_docs = loader.load()

    # Split text into chunks to fit within model context windows
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
except Exception as e:
    print(f"Error loading file: {e}. Please ensure {FILE_NAME} exists.")
    docs = []

# ==============================================================================
# 3. STAGE 1: INITIAL RETRIEVAL (Broad Search)
# ==============================================================================
# Goal: Quickly find a larger set (k=8) of potentially relevant candidates.

# Initialize embedding model (OpenAI is generally higher quality for Stage 1)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(docs, embeddings)

# We retrieve a larger number of documents than we intend to use in the final answer
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

query1 = "What is the modern approach to enterprise security, and how do we protect privileged access and endpoints?"
retrieved_docs = retriever.invoke(query1)

# query2 = "How does Zero Trust Architecture differ from traditional security, and how does CSPM help manage cloud risks?"
# retrieved_docs = retriever.invoke(query2)

# query3 = "How can we use automation to detect software vulnerabilities early and monitor endpoint threats in real time?"
# retrieved_docs = retriever.invoke(query3)

# ==============================================================================
# 4. STAGE 2: RE-RANKING (Precision Sorting)
# ==============================================================================
# Goal: Use an LLM to carefully re-order the retrieved candidates by actual relevance.

# Initialize the LLM (Re-ranker)
MODEL_NAME_CHAT = "groq:llama-3.1-8b-instant"
llm = init_chat_model(MODEL_NAME_CHAT)

# Define the Ranking Prompt
# This asks the LLM to act as a judge and return only the order of indices.
ranking_template = """
You are a helpful assistant. Your task is to rank the following documents from most to least relevant to the user's question.

User Question: "{question}"

Documents:
{documents}

Instructions:
- Think about the relevance of each document to the user's question.
- Return a list of document indices in ranked order, starting from the most relevant.

Output format: comma-separated document indices (e.g., 2,1,3,0,...)
"""
ranking_prompt = PromptTemplate.from_template(ranking_template)

# StrOutputParser - Extract text content from model outputs as a string.
# https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser
# Create the ranking chain
ranking_chain = ranking_prompt | llm | StrOutputParser()

# Format retrieved documents for the LLM prompt
doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
formatted_docs_str = "\n".join(doc_lines)

# print(f"===formatted_docs_str===: {formatted_docs_str}") #NOTE Just for testing 

# Execute re-ranking
ranking_response = ranking_chain.invoke({
    "question": query1, 
    "documents": formatted_docs_str
})

print(f"🔍 Query: {query1}")
print(f"Re-ranking response: {ranking_response}")
print("====================================")

# ==============================================================================
# 5. PARSING & FINAL OUTPUT
# ==============================================================================

# Parse the comma-separated string back into a list of integers
try:
    indices = [int(x.strip()) - 1 for x in ranking_response.split(",") if x.strip().isdigit()]
    # Filter valid indices and reorder the original document list
    reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
except Exception as e:
    print(f"Error parsing re-ranker output: {e}")
    reranked_docs = retrieved_docs  # Fallback to original order

print("\n📊 Final Reranked Results (Top 3):")
for i, doc in enumerate(reranked_docs[:3], 1):
    print(f"\nRank {i}:")
    print(f"---")
    print(f"{doc.page_content}")