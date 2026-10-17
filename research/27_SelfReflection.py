"""==========================================
Self-Reflection RAG using LangGraph
==========================================
Concept:
In Self-Reflection RAG, the agent generates an answer and then "reflects" on it. 
It checks if the answer actually answers the question or if it needs more information. 
If it's not good enough, it loops back to perform another search or refine the answer.

1. Document Grader   ──>  Catches RETRIEVAL Failures (Irrelevant Data) ex. 25d_AgenticRAG_Grader.py
2. Answer Reflector  ──>  Catches GENERATION Failures (Hallucinations / Incomplete Answers) ex. this code
    Catches cases where the retrieved documents were good, but the LLM failed to synthesize them properly 
    (e.g., it missed a key part of a multi-part question, hallucinated details, or gave an overly vague response)
"""

import os
import warnings
from typing import List
from dotenv import load_dotenv

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# Suppress the specific LangChain serialization warning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Language Model
# NOTE test with a different model if the number of attempts is not more than one 
# llm = init_chat_model("openai:gpt-5.4")
llm = init_chat_model("openai:gpt-4o-mini")
# llm = init_chat_model("openai:gpt-4o")

# Initialize Document Loader, Vector Store, and Retriever
INTERNAL_DOCS_PATH = "cybersecurity_data/private_docs.txt"
docs = TextLoader(INTERNAL_DOCS_PATH).load()
embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings_engine)
retriever = vectorstore.as_retriever()


# ### 1. Define the State
# In Pydantic:Objects are instance-based data models.
class RAGReflectionState(BaseModel):
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    reflection: str = ""
    revised: bool = False
    attempts: int = 0


# ### 2. Define the Nodes
# `model_copy` preserves immutability and ensures that LangGraph correctly tracks state transitions 
# from node to node without modifying history in unexpected ways

# #### Node 1: Retriever
def retrieve_docs(state: RAGReflectionState) -> RAGReflectionState:
    print("---RETRIEVER---")
    docs = retriever.invoke(state.question)
    return state.model_copy(update={"retrieved_docs": docs})


# #### Node 2: Generator
def generate_answer(state: RAGReflectionState) -> RAGReflectionState:
    print("---GENERATOR---")
    context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
    prompt = f"""
Use the following context to answer the question:
Context:{context}
Question:{state.question}
"""
    answer = llm.invoke(prompt).content.strip()
    return state.model_copy(update={"answer": answer, "attempts": state.attempts + 1})


# #### Node 3: Reflector (The "Critic")
def reflect_on_answer(state: RAGReflectionState) -> RAGReflectionState:
    print("---REFLECTOR---")
    prompt = f"""
Reflect on the following answer to see if it fully addresses the question. 
State YES if it is complete and correct, or NO with an explanation.

Question: {state.question}

Answer: {state.answer}

Respond like:
Reflection: YES or NO
Explanation: ...
"""
    result = llm.invoke(prompt).content
    # If the answer is good, is_ok evaluates to True
    is_ok = "reflection: yes" in result.lower()
    print(f"🔍 [Reflector Decision] Is the answer sufficient? {'YES' if is_ok else 'NO'}")
    # is_ok = TRUE --> not is_ok = False -> No revision needed -> Graph proceeds to "done"
    return state.model_copy(update={"reflection": result, "revised": not is_ok})


# #### Node 4: Finalizer
def finalize(state: RAGReflectionState) -> RAGReflectionState:
    print("---FINALIZE---")
    return state


# ### 3. LangGraph Workflow Construction
builder = StateGraph(RAGReflectionState)

# Add Nodes
builder.add_node("retriever", retrieve_docs)
builder.add_node("generator", generate_answer)
builder.add_node("reflector", reflect_on_answer)
builder.add_node("done", finalize)

# Set Graph Entry Point START
builder.set_entry_point("retriever")

# Define Workflow Edges
builder.add_edge("retriever", "generator")
builder.add_edge("generator", "reflector")

# Define Conditional Routing based on Reflection Result and Max Attempts
builder.add_conditional_edges(
    "reflector",
    lambda s: "done" if not s.revised or s.attempts >= 2 else "retriever",
    {
        "done": "done",         # Tells LangGraph: "done" string maps to "done" node
        "retriever": "retriever" # Tells LangGraph: "retriever" string maps to "retriever" node
    }
)

builder.add_edge("done", END)

# Compile Graph
graph = builder.compile()

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/SelfReflection.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
# Automatically display/open the image on macOS NOTE _Just for testing purposes
# os.system(f"open {OUTPUT_IMAGE_PATH}")


# ### 4. Run the Agent Workflow
if __name__ == "__main__":
    # user_query = "How does Fastly's proprietary tokenization engine fix the linear computational complexity and latency spikes associated with tracking randomized text strings?"
    # user_query = "What specific cryptographic puzzles does Fastly use to mitigate the O(N) matching complexity caused by legacy regex filters?"
    
    user_query = "How does Fastly leverage the linear matching complexity of static regex filters to isolate origin infrastructure from resource exhaustion?"
    init_state = RAGReflectionState(question=user_query)
    
    print(f"Starting workflow for query: '{user_query}'\n")
    result = graph.invoke(init_state)

    print("\n=================== RESULTS ===================")
    print("🧠 Final Answer:\n", result["answer"])
    print("\n🔁 Reflection Log:\n", result["reflection"])
    print("\n🔄 Total Attempts:", result["attempts"])
    print("===============================================")