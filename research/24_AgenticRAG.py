"""### Agentic RAG 🤖 What is Agentic RAG? ###
Agentic RAG stands for Agentic Retrieval-Augmented Generation — an advanced version of RAG where instead of a static, 
one-shot LLM response,the system uses an agent that:
- reasons,
- plans,
- retrieves,
- uses tools,
- and even retries or reflects
to generate better, more grounded answers.

================================================================================
This script introduces **Agentic RAG (Retrieval-Augmented Generation)** implemented 
via LangGraph. Unlike traditional, linear RAG pipelines that execute 
a rigid "retrieve-then-generate" script in a one-shot fashion, Agentic RAG shifts 
the workload into a stateful, graph-driven execution canvas. The system uses an 
orchestration agent that can reason, plan, use tools, and loop between nodes dynamically.


1. WEB DOCUMENT PREPROCESSING: Uses `WebBaseLoader` to pull live enterprise cyber security 
   and bot management blogs from cloud vendor technical guides (Cloudflare, AWS, Indusface).
2. VECTORIZATION LAYER: Splits the raw HTML text extractions into 500-character blocks 
   and builds a local FAISS database with OpenAI's `text-embedding-3-small` model.
3. PYDANTIC STATE DESIGN (`RAGState`): Establishes a Pydantic-backed state schema tracking 
   the raw question string, a dynamic list of recovered `Document` objects, and the 
   final response payload. This ensures data validation at node boundaries.
4. STATEGRAPH BUILD: Registers functional worker nodes (`retriever` and `responder`) 
   and compiles them into a workflow graph ending at the terminal `END` state.
5. EXECUTION: Runs a complex security query forcing the model to synthesize text 
   regarding client-side challenges and traditional WAF logic from the context.
================================================================================
"""

import os
import warnings
import logging

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END


# Suppress standard Python and Transformer warnings/progress logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
# llm=init_chat_model("openai:gpt-4o")
llm=init_chat_model("openai:gpt-4o-mini")

# -----------------------------
# 1. Document Preprocessing
# -----------------------------
urls = [
    # Cloudflare's breakdown of WAF ML models and threat intelligence
    "https://blog.cloudflare.com/tag/waf/", 
    # AWS technical guide on configuring WAF rules specifically for aggressive AI scrapers
    "https://aws.amazon.com/blogs/networking-and-content-delivery/how-to-manage-ai-bots-with-aws-waf-and-enhance-security/",
    # Indusface enterprise analysis of bot mitigation mechanics
    "https://www.indusface.com/blog/top-bot-management-software/"
]

# https://reference.langchain.com/python/langchain-community/document_loaders/web_base/WebBaseLoader 
# WebBaseLoader document loader integration
loaders = [WebBaseLoader(url) for url in urls]
docs = []
for loader in loaders:
    docs.extend(loader.load())

## Recursive character text ssplitter an vectorstore
# https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# https://reference.langchain.com/python/langchain-openai/embeddings/base/OpenAIEmbeddings
# https://developers.openai.com/api/docs/models/text-embedding-3-small
# OpenAI embedding model integration.
EMBEDDING_MODEL = "text-embedding-3-small"
embedding = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    dimensions=512
)

vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()

# NOTE Just for testing
# retriever.invoke("What is WAF")

# -----------------------------
# 2. Define RAG State
# -----------------------------
class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""

# -----------------------------
# 3. LangGraph Nodes
# -----------------------------

def retrieve_docs(state: RAGState) -> RAGState:
    docs = retriever.invoke(state.question)
    return RAGState(question=state.question, retrieved_docs=docs)

def generate_answer(state: RAGState) -> RAGState:    
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {state.question}"
    response = llm.invoke(prompt)
    return RAGState(question=state.question, retrieved_docs=state.retrieved_docs, answer=response.content)

# -----------------------------
# 4. Build LangGraph
# -----------------------------

builder = StateGraph(RAGState)

builder.add_node("retriever", retrieve_docs)
builder.add_node("responder", generate_answer)

# `set_entry_point` is same as builder.add_edge(START, "retriever")
builder.set_entry_point("retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)

OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/24_AgenticRAG.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# 2. Automatically display/open the image on macOS
os.system(f"open {OUTPUT_IMAGE_PATH}")


# -----------------------------
# 5. Run the Agentic RAG
# -----------------------------

if __name__ == "__main__":
    # This question forces the RAG to synthesize data regarding WAF capabilities, 
    # behavioral analysis, and computational client challenges from the articles.
    user_question = (
        "What are the primary differences between traditional rule-based WAF filtering "
        "and behavioral-based Bot Management? Specifically, how do modern systems use "
        "silent client-side challenges (like proof-of-work) to stop evasive scrapers?"
    )
    initial_state = RAGState(question=user_question)
    final_state = graph.invoke(initial_state)

    print("\n✅ Final Answer:\n", final_state['answer'])

print("\n✅ Retrieved Documents:")
for doc in final_state['retrieved_docs']:
    print(f"- {doc.metadata.get('source', 'unknown source')}: {doc.page_content[:200]}...") 
    