"""
### 🤖 Advanced ReAct with LangGraph - Multi-Source Corporate Routing

Goal: 
    Demonstrates how a ReAct agent dynamically evaluates semantic tool descriptions 
to pick between proprietary internal docs, technical research notes, or global academic indexes.

In 25a, we built our foundational ReAct agent. Think of it like creating a simple personal assistant. 
It has a single, fixed memory, and when you ask a question, it only knows how to look at two specific, 
hardcoded web sources (Fastly and Wikipedia). Everything is written manually step-by-step. 
It works great for simple, specific questions, but it doesn't scale well if you add dozens of data sources.

In 25b, we upgraded our agent into a multi-source enterprise traffic router. 
Here are the 3 major architectural improvements we introduced:

🏭 1. From Hardcoded Scripts to Tool Factories:
    In 25a: We created each retriever tool manually, line by line.  
    In 25b: We built a factory function (make_retriever_tool_from_text). 
    Now, we can turn any text file or document folder into an AI-ready tool in a single line of code. 
    
🔀 2. Multi-Domain Diversification (Private vs. Public Data):
    In 25a: The agent only searched public web information.  
    In 25b: We gave the agent a mixed toolkit—combining private internal docs, internal AI research notes, Wikipedia, 
    and real ArXiv academic research papers.  
    
🎯 3. Semantic Routing via Descriptions:
    In 25a: The agent just picked between a web tool and Wikipedia.  
    In 25b: We taught the agent how to reason about semantic descriptions. When asked a complex, multi-part question, 
        the LLM reads each tool's description and autonomously decides: "I need to check internal tech docs for WAF configs, 
        but use ArXiv for machine learning papers.
"""

import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"


# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.graph import END, StateGraph


from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader


from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage

# Initialize Environment & LLM
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# llm = init_chat_model("openai:gpt-4o")
llm = init_chat_model("openai:gpt-4o-mini")

# -----------------------------------------------
# 1. Generic Factory to Create Retriever Tools
# -----------------------------------------------
# Dynamically creates file dependencies if missing, embeds the content using OpenAIEmbeddings, 
# builds a FAISS vectorstore, and returns a fully configured Tool object in a single function call
def make_retriever_tool_from_text(file_path: str, name: str, desc: str) -> Tool:
    # Ensure local mockup file exists to avoid crashes
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Mock context data for {name}. Mentioning transformer variants and custom system architectures.")
            
    docs = TextLoader(file_path, encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(chunks, embedding)
    retriever = vs.as_retriever()

    def tool_func(query: str) -> str:
        print(f"📚 Using internal tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)
    
    return Tool(name=name, description=desc, func=tool_func)

# Instantiating Local Tools
# Combines 4 distinct tools: 
#   two internal file retrievers (InternalTechDocs, InternalResearchNotes),
#    Wikipedia, and Tavily as ArXiv for academic papers.

INTERNAL_DOCS_PATH = "cybersecurity_data/public_docs.txt"
RESEARCH_NOTES_PATH = "cybersecurity_data/private_docs.txt"

internal_tool_1 = make_retriever_tool_from_text(
    INTERNAL_DOCS_PATH,
    "InternalTechDocs",
    "Search internal tech documents for proprietary architecture, WAF configurations, and infrastructure methods."
)

internal_tool_2 = make_retriever_tool_from_text(
    RESEARCH_NOTES_PATH,
    "InternalResearchNotes",
    "Search internal research notes for AI experimental results, transformer variants, and agent designs."
)

# ---------------------------------------
# 1. Wikipedia Tool Tool Configuration
# ---------------------------------------
# Initialize the underlying Wikipedia API Wrapper
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

# Define a custom function to intercept the execution, add your print statement, safely catch network/API errors
def wiki_tool_func(query: str) -> str:
    print("🌐 Using Wikipedia tool")
    try:
        return wiki_api.run(query)
    except Exception as e:
        # Prevents JSONDecodeError or HTTP errors from crashing the agent graph
        return f"Wikipedia Lookup Error: Could not retrieve summary for '{query}'. Error details: {e}"
        
# Re-create the wiki_tool using the standard Tool wrapper
wiki_tool = Tool(
    name="Wikipedia",
    description="A wrapper around Wikipedia. Use this tool to fetch general world knowledge from Wikipedia..",
    func=wiki_tool_func
)

# ---------------------------------------
# 2. Tavily as ArXiv Tool Configuration
# ---------------------------------------
# Tavily restricted specifically to arxiv.org
arxiv_via_tavily = TavilySearch(
    name="ArxivSearch", # Keeps the tool name aligned with your prompts!
    description="Search arXiv.org for recent academic papers on machine learning and security.",
    max_results=1,
    include_domains=["arxiv.org"]
)

# ---------------------------------------
# 3. Compile Graph with Multi-Source Tools
# ---------------------------------------
# Pass a strict system prompt to force multi-tool validation
sys_prompt = (
    "You are an enterprise research agent. For EVERY query, you MUST invoke and "
    "cross-reference ALL four tools: 'InternalTechDocs', 'InternalResearchNotes', "
    "'Wikipedia', and 'ArxivSearch'. Do not generate the final answer until you have "
    "gathered observations from all four sources."
)

# All Tools Loaded in the Toolkit
tools = [wiki_tool, arxiv_via_tavily, internal_tool_1, internal_tool_2]
# print("\n🛠️  Registered Agent Tools:")
# for tool in tools:
#     # Works for both custom functions and prebuilt objects
#     print(f"  • [{tool.name}]: {tool.description[:70]}...")
# print("-" * 50 + "\n")
react_node = create_agent(llm, tools, system_prompt=sys_prompt)

# NOTE : below helps with debugging but is verbose
# react_node = create_agent(llm, tools, debug=True)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

builder = StateGraph(AgentState)
builder.add_node("react_node", react_node)
builder.set_entry_point("react_node")
builder.add_edge("react_node", END)

# Instantiate in-memory checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ---------------------------------------
# 3. Run Complex Cross-Cutting Query
# ---------------------------------------
if __name__ == "__main__":
    #  Questions  (All Docs and Arvix): 
    config = {"configurable": {"thread_id": "session_1"}}

    user_question = (
        "Perform a multi-source audit on WAF technology:\n"
        "1. Query 'InternalResearchNotes' and 'InternalTechDocs' for Fastly WAF and legacy regex limitations.\n"
        "2. Query 'Wikipedia' for the general technical definition of 'Web Application Firewall'.\n"
        "3. Query 'ArxivSearch' for recent academic papers on 'machine learning web application firewall'.\n"
        "Synthesize the results from ALL sources into distinct sections."
    )    
    state = {"messages": [HumanMessage(content=user_question)]}
    result = graph.invoke(state, config=config)
    

    print("\n✅ Final Answer:\n", result["messages"][-1].content)