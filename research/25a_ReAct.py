"""
### 🤖 Implement ReAct with LangGraph - Basic Framework
ReAct (Reasoning + Acting) is a framework where an LLM:
- Reasons step-by-step (e.g., decomposes questions, makes decisions)
- Acts by calling tools like search, calculators, or retrievers

✅ Think → Retrieve → Observe → Reflect → Final Answer

React_agent_basic (The Core Framework) with just 2 sources
    Data Sources:Just WebBaseLoader (from a WAF blog) and the standard WikipediaQueryRun.

Goal: Teach the standard ReAct framework loop using simple, predictable, public APIs.
Focus on how create_agent binds a chat model to tools and executes a simple Think --> Act --> Observe loop.
Demonstrates the fundamental ReAct loop (Think --> Act --> Observe) using a prebuilt LangGraph agent linked 
    to a clean vector retriever and Wikipedia.

Key Advantages of using create_agent():
1. Native System Prompts & Steering:
   Accepts `system_prompt` directly to enforce rules across steps without 
   manually manipulating message arrays.

2. Advanced Error Recovery:
   Includes built-in retry logic that feeds tool errors back to the LLM 
   so it can self-correct parameter issues automatically.

3. Minimal Boilerplate:
   Replaces ~25 lines of manual graph setup (defining State, nodes, edges, 
   and tools_condition) with a single, standard call.    
"""


import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Set a custom User-Agent identifying your application
os.environ["USER_AGENT"] = "Agentic-RAG-Cookbook/1.0 (contact: ash@codeaiwashnaiku.com)"

from langchain.agents import create_agent
from langgraph.graph import END, StateGraph

from langchain_core.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage

# Initialize Environment & LLM
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# llm = init_chat_model("openai:gpt-4o")
llm = init_chat_model("openai:gpt-4o-mini")

# --------------------------
# 1. Create Retriever Tool and Wikipedia Tool
# --------------------------
urls = [
    "https://www.fastly.com/blog/ua-spoofing-101-detection-defense-with-fastlys-next-gen-waf",
    "https://aws.amazon.com/blogs/networking-and-content-delivery/how-to-manage-ai-bots-with-aws-waf-and-enhance-security/",
]

# WebBaseLoader document loader integration
# https://reference.langchain.com/python/langchain-community/document_loaders/web_base/WebBaseLoader
docs = []
for url in urls:
    docs.extend(WebBaseLoader(url).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

def retriever_tool_func(query: str) -> str:
    print("📚 Using RAGRetriever tool")
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# Retriever tool
retriever_tool = Tool(
    name="RAGRetriever",
    description="Use this tool to fetch relevant knowledge base info about web security and WAFs.",
    func=retriever_tool_func
)

# Wikipedia tool
# Initialize the underlying Wikipedia API Wrapper
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)

# Define a custom function to intercept the execution and add your print statement
def wiki_tool_func(query: str) -> str:
    print("🌐 Using Wikipedia tool")  # Your new custom print statement
    return wiki_api.run(query)

# Re-create the wiki_tool using the standard Tool wrapper
wiki_tool = Tool(
    name="Wikipedia",
    description="A wrapper around Wikipedia. Use this tool to fetch general world knowledge from Wikipedia..",
    func=wiki_tool_func
)

# ----------------------------
# 2. Define the Agent Node & Tools
# ----------------------------
# All Tools Loaded in the Toolkit
tools = [retriever_tool, wiki_tool]
print("\n🛠️  Registered Agent Tools:")
for tool in tools:
    # Works for both custom functions and prebuilt objects
    print(f"  • [{tool.name}]: {tool.description[:70]}...")
print("-" * 50 + "\n")


# Create the native Langgraph react agent execution node
# create_agent automatically gives the LLM the ability to think, pick tools, 
# check results, and loop on its own until it finishes the task
react_node = create_agent(llm, tools)
# NOTE : below helps with debugging but is verbose
# react_node = create_agent(llm, tools, debug=True)

# --------------------------
# 3. LangGraph Agent State
# --------------------------
# Sequence is a broader interface that includes anything you can iterate over 
# as list, tuple, or custom sequence objects
# AnyMessage is ideal when working with standard Chat Models, as it restricts state strictly to expected conversation turns.
# BaseMessage is helpful if you build custom message classes or extend standard message types, as AnyMessage might throw static type warnings for non-standard subclasses.

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --------------------------
# 4. Build LangGraph Graph
# --------------------------
builder = StateGraph(AgentState)

builder.add_node("react_node", react_node)
builder.set_entry_point("react_node")
builder.add_edge("react_node", END)


# Instantiate in-memory checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Generate and automatically show system diagram layout on execution
OUTPUT_IMAGE_FOLDER = "Image_PNGs"
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
OUTPUT_IMAGE_PATH = OUTPUT_IMAGE_FOLDER + "/25a_ReAct.png"
graph.get_graph().draw_mermaid_png(output_file_path=OUTPUT_IMAGE_PATH)    
os.system(f"open {OUTPUT_IMAGE_PATH}")


# --------------------------
# 5. Run the ReAct Agent
# --------------------------
if __name__ == "__main__":
    # Define a thread ID to identify this specific chat session
    config = {"configurable": {"thread_id": "session_1"}}

    user_query = "What is User Agent spoofing as per Fastly blog and how does Wikipedia describe XSS attacks?"
    state = {"messages": [HumanMessage(content=user_query)]}

    # Pass config as the second parameter
    result = graph.invoke(state, config=config)

    print("\n✅ Final Answer:\n", result["messages"][-1].content)

    # Subsequent call on the SAME thread_id will preserve prior messages automatically
    follow_up = {"messages": [HumanMessage(content="Summarize your previous answer in one sentence.")]}
    follow_up_result = graph.invoke(follow_up, config=config)

    print("\n✅ Follow-up Answer:\n", follow_up_result["messages"][-1].content)