"""
    Corrective RAG (CRAG) with Custom Decision-Making & Tool Creation

* Main Flow: Executes a self-correcting RAG pipeline that evaluates retrieved 
  context before generating an answer to eliminate hallucinations.

* How Grading Drives Decisions: The `grade_documents` node checks relevance. If 
  relevant ("yes"), it routes to `generate`; if irrelevant ("no"), it routes 
  to `rewrite` to refine the user query or `fallback` after max retries.

* `create_retriever_tool()`: Wraps the FAISS vector database into a structured 
  tool so the agent node can dynamically discover and query internal docs.

1. TOOL DEFINITION WITH `create_retriever_tool()`:
   - Function: Wraps a raw Vector Store Retriever (FAISS) into a standardized 
     LangChain Tool object.
   - Purpose: Exposes the retriever to the LLM with a clear name and natural language 
     description. This allows the model to dynamically decide when and how to 
     query the internal vector store during tool invocation.

2. CUSTOM GRAPH ARCHITECTURE vs. HIGH-LEVEL `create_agent()`:
   - Why Custom Graph Nodes? 
     High-level agent constructors like `create_agent()`  
     provide prebuilt execution loops optimized for standard tool-calling. 
     However, they lack native mechanisms to intercept retrieved content for quality grading.
   - Self-Correction Mechanism (Corrective RAG / CRAG): Manually builds a custom 
     `StateGraph` with explicit nodes (`agent`, `retrieve`, `grade_documents`, `rewrite`, `generate`, `fallback`). 
   - Interception & Routing:
     Constructing the flow manually allows the `grade_documents` node to evaluate 
     retrieved documents immediately after retrieval. If documents are non-relevant, 
     the flow conditionally reroutes to `rewrite` (query refinement) or `fallback` 
     rather than returning potentially hallucinated answers.

"""

import os
from typing import Annotated, Literal, Sequence
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

# Load environment configs
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Core LangChain Framework
from langchain_classic import hub
from langchain.chat_models import init_chat_model
from langchain_core.tools import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# LangChain Ecosystem Components
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# LangGraph Architecture
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# ==============================================================================
# 1. INITIAL BASELINE & DATA PREPROCESSING
# ==============================================================================

# Common embedding engine instance configurations
embeddings_engine = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Data Ingestion: WebSecurity WAF & Bot Protection ---
waf_urls = [
    "https://www.fastly.com/blog/ua-spoofing-101-detection-defense-with-fastlys-next-gen-waf",
    "https://www.fastly.com/blog/credential-stuffing-attacks-vs-brute-force-attacks-what-is-the-difference",
    "https://www.fastly.com/blog/what-is-cve-2026-23869-react-server-components-security-alert",
    "https://www.fastly.com/blog/ddos-in-december-2025"
]

waf_docs = [item for url in waf_urls for item in WebBaseLoader(url).load()]
waf_splits = splitter.split_documents(waf_docs)
vectorstore_waf = FAISS.from_documents(documents=waf_splits, embedding=embeddings_engine)
retriever_waf = vectorstore_waf.as_retriever()

# https://reference.langchain.com/python/langchain-core/tools/retriever/create_retriever_tool
retriever_tool_wafbot = create_retriever_tool(
    retriever_waf,
    "retriever_waf_db_blog",
    "Search and run information about WAF & BOT protection"
)


# Active Toolkit registration
tools = [retriever_tool_wafbot]

# ==============================================================================
# 2. CORE LLM & SCHEMAS DEFINITION
# ==============================================================================

# Consolidated Global Model definition to limit repeated construction calls
shared_llm = init_chat_model("groq:qwen/qwen3.6-27b")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int

class DocumentGrade(BaseModel):
    """Binary score for relevance checks."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# ==============================================================================
# 3. WORKFLOW NODES & ROUTERS INTERPOLATION
# ==============================================================================

# Invokes the model bound with tools to decide whether to call a retrieval tool or respond directly.
def agent(state: AgentState):
    print("---CALL AGENT---")
    messages = state["messages"]
    model_with_tools = shared_llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Returns a friendly error message when the graph exceeds max retry attempts without finding relevant context.
def fallback(state: AgentState):
    print("---FALLBACK: MAX RETRIES EXCEEDED---")
    error_msg = (
        "I'm sorry, but I couldn't find any relevant context in the database "
        "to answer your question accurately."
    )
    return {"messages": [HumanMessage(content=error_msg)]}    

# Evaluates retrieved context relevance; routes to 'generate' if valid, or to 'rewrite'/'fallback' if irrelevant.
def grade_documents(state: AgentState) -> Literal["generate", "rewrite", "fallback"]:
    print("---CHECK RELEVANCE---")
    
    # Extract retries from state (defaults to 0 if not set yet)
    # https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/with_structured_output
    # Model wrapper that returns outputs formatted to match the given schema.
    retries = state.get("retry_count", 0)
    # https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/with_structured_output
    llm_with_tool = shared_llm.with_structured_output(DocumentGrade)
    
    prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        )

    chain = prompt | llm_with_tool
    messages = state["messages"]
    
    question = messages[0].content
    docs_content = messages[-1].content  # Extract text from latest ToolMessage response context
    
    scored_result = chain.invoke({"question": question, "context": docs_content})
    score = scored_result.binary_score
    
    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    elif retries >= 2:
        print("---DECISION: DOCS NOT RELEVANT (MAX RETRIES REACHED)---")
        return "fallback"        
    else:
        print(f"---DECISION: DOCS NOT RELEVANT (Attempt {retries + 1})---")
        print(f"Confidence score evaluated as: {score}")
        return "rewrite"


# Uses the retrieved context to synthesize a concise final answer for the user's question.
def generate(state: AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs_content = messages[-1].content

    # NOTE Just for testing 
    # print(f"\n📄 For question {question} [Retrieved Context ]:\n{docs_content}\n")

    prompt_template = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer:"
    )
    rag_chain = prompt_template | shared_llm | StrOutputParser()
    
    response = rag_chain.invoke({"context": docs_content, "question": question})
    return {"messages": [response]}

# Rewrites and optimizes the initial user question to improve search results on subsequent retrieval attempts.
def rewrite(state: AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    retries = state.get("retry_count", 0)

    rewrite_message = [
        HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:\n -------\n{question}\n -------\nFormulate an improved question: """)
    ]
    
    response = shared_llm.invoke(rewrite_message)
    return {"messages": [response],
        "retry_count": retries + 1
    }

# ==============================================================================
# 4. GRAPH ARCHITECTURE COMPILATION
# ==============================================================================

workflow = StateGraph(AgentState)

# Add Nodes
# ToolNode(tools) - Automatically reads the output of an LLM call 
# (which contains a tool call request, like retriever_waf_db_blog(query=...)), 
# matches it to the right tool in your tools list
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode(tools))  # Maps natively over our registered tools list
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)
workflow.add_node("fallback", fallback)

# Define Core Execution Pathways
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Conditional router from retrieve
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "fallback": "fallback"
    }
)


workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Save the file as a PNG
OUTPUT_IMAGE_PATH = "Image_PNGs/25c_AgenticRAG_Grader.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# ==============================================================================
# 5. DIAGNOSTIC DIAGRAM PRODUCTION & INVOCATIONS
# ==============================================================================
config = {"configurable": {"thread_id": "session_1"}}
# Execution runs
print("\n🛡️ Executing WebSecurity Query 1:")
# Question 1: User-Agent Spoofing & Fingerprinting Heuristics
graph.invoke({"messages": [HumanMessage(content="How does the Fastly Next-Gen WAF move beyond signature-based regex filters , and what role do its behavioral signals and TLS/JA3 fingerprinting play in detecting these hidden bots?")]}, config=config)

import time
time.sleep(10)

print("\n🛡️ Executing WebSecurity Query 2:")
# Question 2: Threat Taxonomy (Credential Stuffing vs. Brute Force)
graph.invoke({"messages": [HumanMessage(content="According to the Fastly threat breakdown, what is the core structural difference between a brute force attack and a credential stuffing attack")]}, config=config)

time.sleep(10)

print("\n🛡️ Executing WebSecurity Query (Out-of-Domain / Forces 'no'):")
graph.invoke({
    "messages": [
        HumanMessage(
            content="How does Fastly Next-Gen WAF use quantum computing and nuclear fusion to stop botnet attacks?"
        )
    ]
}, config=config)