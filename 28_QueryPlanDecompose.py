"""
### 🧠 What is Query Planning and Decomposition?
Query Planning and Decomposition is a technique where a complex user query is broken down into simpler sub-questions or tasks.

It's like reverse-engineering a question into manageable steps before answering.

🧠  Planner: Breaks query into sub-questions.  
    Retriever: Loops over sub-questions, retrieves relevant context documents for each, 
                and merges all documents into one single list (all_docs).  
    Generator: Takes the entire combined context and the original user query, then generates a single, fully synthesized answer.  

13_QueryDecomposition.py does NOT use StateGraph
StateGraph, it is trivial to extend. If a sub-question retrieves no results, or if the answer fails self-reflection, you can easily add conditional edges to re-try or loop back.

Unlike 13_QueryDecomposition.py this code aggregates all context before answering. Complex queries often require connecting dots across different sub-questions

[CoT-RAG (26_COTRag.py)]
Complex Query ──> Planner (Compresses into 1 Unified Search String) ──> Single Search (Tavily) ──> Final Synthesis

[Query Decomposition (This file)]
Complex Query ──> Planner (Splits into 2-3 Distinct Sub-Questions) ──>
     Multiple Sequential Searches (FAISS) ──> Consolidated Synthesis
"""

import os
import warnings
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
# Suppress the specific LangChain serialization warning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader,WebBaseLoader
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = init_chat_model("openai:gpt-4o-mini")

# ----------------------------
# 1. Load and Embed Documents
# ----------------------------

urls = [
    "https://www.fastly.com/blog/ua-spoofing-101-detection-defense-with-fastlys-next-gen-waf",
    "https://www.fastly.com/blog/credential-stuffing-attacks-vs-brute-force-attacks-what-is-the-difference",
    "https://www.fastly.com/blog/what-is-cve-2026-23869-react-server-components-security-alert",
    "https://www.fastly.com/blog/ddos-in-december-2025"
]
docs = []
for url in urls:
    docs.extend(WebBaseLoader(url).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# ----------------------------
# 2. State Schema
# ----------------------------
class RAGState(BaseModel):
    question: str
    sub_questions: List[str] = []
    retrieved_docs: List[Document] = []
    answer: str = ""


# ----------------------------
# 3. Nodes
# ----------------------------

## a. Query Planner: splits input question
def plan_query(state: RAGState) -> RAGState:
   
    prompt = f"""
Break the following complex question into 2-3 sub-questions:

Question: {state.question}

Sub-questions:
"""
    result = llm.invoke(prompt)
    sub_questions = [line.strip("- ").strip() for line in result.content.strip().split("\n") if line.strip()]
    return RAGState(question=state.question, sub_questions=sub_questions)

## b. Retrieve documents for each sub-question
def retrieve_for_each(state: RAGState) -> RAGState:
    all_docs = []
    for sub in state.sub_questions:
        docs = retriever.invoke(sub)
        all_docs.extend(docs)
    return RAGState(question=state.question, sub_questions=state.sub_questions, retrieved_docs=all_docs)

## c. Generate final answer
def generate_final_answer(state: RAGState) -> RAGState:
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"""
Use the context below to answer the question.
Context:{context}
Question: {state.question}
"""
    
    answer = llm.invoke(prompt).content
    return RAGState(question=state.question, sub_questions=state.sub_questions, retrieved_docs=state.retrieved_docs, answer=answer)

# ----------------------------
# 4. Build LangGraph
# ----------------------------
builder = StateGraph(RAGState)

builder.add_node("planner", plan_query)
builder.add_node("retriever", retrieve_for_each)
builder.add_node("generator", generate_final_answer)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "generator")
builder.add_edge("generator", END)

graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/QueryPlanningDecomposition.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
os.system(f"open {OUTPUT_IMAGE_PATH}")

# ----------------------------
# 5. Run the pipeline
# ----------------------------
if __name__ == "__main__":
    user_query = "How does the Fastly Next-Gen WAF move beyond signature-based regex filters to unmask User-Agent spoofing, and what role do its behavioral signals and TLS/JA3 fingerprinting play in detecting these hidden bots?"
    # user_query = "According to the Fastly threat breakdown, what is the core structural difference between a brute force attack and a credential stuffing attack, and why is credential stuffing considered a surgical subset?"
    # user_query = "Explain the mechanics of the high-severity CVE-2026-23869 DoS vulnerability in React Server Components, and describe how Fastly's automated DDoS protection handled 'The Grinch' botnet attack on Christmas Day 2025."
    initial_state = RAGState(question=user_query)
    final_state = graph.invoke(initial_state)
    print(final_state)

    print("\n🔍 Sub-questions:")
    for q in final_state['sub_questions']:
        print("-", q)

    print("\n✅ Final Answer:\n", final_state['answer'])
    