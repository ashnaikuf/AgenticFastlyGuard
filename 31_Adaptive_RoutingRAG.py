"""
================================================================================
ADAPTIVE RAG (Routing-Driven Retrieval)
================================================================================
This script implements Adaptive RAG, a dynamic retrieval strategy that inspects 
user query intent up front to route requests along optimal execution paths, 
preventing wasteful vector database lookups or unnecessary external API calls.

THE TWO ROUTING PATHWAYS:
1. Vector Store (vectorstore): For domain-specific questions covered in local 
   knowledge bases (e.g., proprietary documentation).
2. Web Search (web_search): For real-time updates, current events, or general 
   knowledge missing from internal stores.

GRAPH EXECUTION & EDGE FLOW:
  START 
    │
    ▼
[ router_node ]  ---> Evaluates query intent via structured output (RouteQuery)
    │
    ├── ( if route == "vectorstore" ) ──> [ retrieve ] ──────┐
    │                                                        │
    └── ( if route == "web_search" )  ──> [ web_search ] ────┴─> [ generate ] ──> END

DETAILED NODE & FUNCTION MAPPING:
1. START ---> router_node
   - Function: router_node(state)
   - Action: Uses structured LLM output (RouteQuery) to classify the question's 
     intent as "vectorstore" or "web_search".

2. router_node ---> retrieve OR web_search (Conditional Edge)
   - Branch A: "vectorstore" ---> retrieve(state)
     Pulls relevant source chunks from the local FAISS database using the retriever.
   - Branch B: "web_search" ---> web_search(state)
     Executes a live query via TavilySearchResults to fetch external context.

3. retrieve OR web_search ---> generate
   - Function: generate(state)
   - Action: Synthesizes a response using solely the context gathered from 
     either local documents or web results.

4. generate ---> END
   - Final state output returned to the user.
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
# Suppress the specific LangChain serialization warning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ### ----------------- 1. Setup Vector Store -----------------###
# Creating a small local knowledge base for LangGraph specific info.
docs = [
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs."),
    Document(page_content="Adaptive RAG routes queries to web search or vector stores based on intent."),
    Document(page_content="Nodes in LangGraph represent functions, and edges represent the flow of state.")
]

embedding=OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Instantiate once at the script/module level
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(k=2)

# ### ----------------- 2. Define the State ----------------- ###
# TypedDict is optimized for LangGraph's internal state management, 
# BaseModel (Pydantic) is required by LangChain for LLM Structured Output validation.
# https://typing.python.org/en/latest/spec/typeddict.html
class State(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    route: str # Tracks the decision made by the router

# ### ----------------- 3. Define the Router (The Key Component) ----------------- ###
# We use structured output to force the LLM to choose a path.
# https://pydantic.dev/docs/validation/dev/concepts/models/
class RouteQuery(BaseModel):
    """Route a user query to the most appropriate datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(description="Given a user question, choose whether to route it to 'web_search' or 'vectorstore'."
    )

def router_node(state: State):
    print(" ---ROUTING---")

    structured_llm = llm.with_structured_output(RouteQuery)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert at routing a user question to a vectorstore or web search. "
        "The vectorstore contains documents related to LangGraph. "
        "Use web_search for questions about recent events or general knowledge. "
        "Question: {question}"
    )
    
    router_chain = prompt | structured_llm
    result = router_chain.invoke({"question": state["question"]})

    # NOTE another approach - Directly specify the destination node in the Command return! 
    # destination = result.datasource if result.datasource == "web_search" else "retrieve"
    # return Command(goto=destination, update={"route": destination})

    return {"route": result.datasource}


# ----------------------------------------
# route_decision is a routing function that runs between nodes. Used with add_conditional_edges
def route_decision(state: State) -> Literal["retrieve", "web_search"]:
    """Returns the name of the next node based on the router's decision."""
    if state["route"] == "web_search":
        return "web_search"
    return "retrieve"


# ### ----------------- 4. Define Functional Nodes ----------------- ###

# #### Node: Retrieve from Vector Store
def retrieve(state: State):
    print(" ---RETRIEVING FROM VECTOR STORE---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

# #### Node: Web Search
def web_search(state: State):
    print(" ---SEARCHING THE WEB---")
    search_results = search_tool.invoke(state["question"])
    joined_results = "\n".join([res["content"] for res in search_results])
    web_doc = Document(page_content=joined_results)
    return {"documents": [web_doc]}

# #### Node: Generate Answer
def generate(state: State):
    print(" ---GENERATING---")
    prompt = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the provided context:\n{context}\nQuestion: {question}"
    )
    context = "\n\n".join([d.page_content for d in state["documents"]])
    chain = prompt | llm
    response = chain.invoke({"question": state["question"], "context": context})
    return {"answer": response.content}

# ### ----------------- 5. Build the Graph ----------------- ###
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("router_node", router_node)
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# Define Entry Point and Conditional Edges
workflow.add_edge(START, "router_node")

# Graph setup
workflow.add_conditional_edges(
    "router_node",
    route_decision,
    {
        "retrieve": "retrieve",
        "web_search": "web_search"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# Generate and automatically show system diagram layout on execution
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/31_Adaptive_RoutingRAG.png"
app.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# ### 6. Run Adaptive RAG ###
# Case A: Query routes to Vector Store
print("---Case A: Internal Query---")
res1 = app.invoke({"question": "Explain what a node is in LangGraph."})
print(f"Final Answer: {res1['answer']}")

# Case B: Query routes to Web Search
print("\n---Case B: Trending/General Query---")
res2 = app.invoke({"question": "What is the weather in New York today?"})
print(f"Final Answer: {res2['answer']}")



"""
ARCHITECTURAL COMPARISON: ReAct Tool-Calling Agent vs. Adaptive RAG Router
================================================================================
While both scripts dynamically choose execution paths based on the query,
they operate on two fundamentally different architectural paradigms:

1. REACTION & LOOPS (ReAct Agent - 23a_AgentsArch_MathTools.py):
   - HOW IT WORKS: The LLM acts as an autonomous controller in an iterative 
     loop (Reason + Act + Observe) to select, execute, and analyze tools.
   - EXECUTION PATTERN: Dynamic and multi-step. It can run zero, one, or 
     multiple tool calls in a single user turn.

2. INTENT GATEKEEPING (Adaptive RAG Router - 31_Adaptive_RoutingRAG.py):
   - HOW IT WORKS: An upfront router node uses structured LLM output to 
     classify query intent once before any retrieval begins.
   - EXECUTION PATTERN: Deterministic and single-pass. It branches early, 
     executes the chosen node, generates an answer, and ends.

--------------------------------------------------------------------------------
- ReAct (23a): Best for multi-step reasoning and dynamic tool chaining.
- Adaptive RAG (31): Best for routing queries upfront to reduce latency and API costs.
================================================================================
"""