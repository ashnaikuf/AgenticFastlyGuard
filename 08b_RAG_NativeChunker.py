"""
================================================================================
This script contrasts the custom threshold builder (`08a`) by showcasing 
LangChain's built-in **`SemanticChunker`** class. It eliminates manual distance 
looping by leveraging statistical anomalies to auto-determine chunk boundaries.

1. INSTANTIATION: Configures LangChain's experimental `SemanticChunker`. 
   Unlike character-count splitters, this utilizes OpenAI's embedding model 
   directly to determine distance gaps between consecutive sentences.
2. STATISTICAL BREAKPOINTS: Configures a `standard_deviation` threshold strategy. 
   When the distance between adjacent sentences deviates beyond `1.25` standard 
   deviations from the average, a semantic breakpoint is automatically triggered.
3. RAG PIPELINE EXECUTION: Embeds the resulting native semantic chunks into a 
   FAISS vector space and passes the retrieved contextual chunks into an LLM 
   for high-accuracy query extraction.
================================================================================
"""

import os
import warnings
from dotenv import load_dotenv

# LangChain Core & Models
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

# Vector Store & Native Chunker
from langchain_community.vectorstores import FAISS
#  DeprecationWarning: `langchain-experimental` is being sunset and is no longer actively maintained.
#  See https://github.com/langchain-ai/langchain-experimental/issues/87 for details.
from langchain_experimental.text_splitter import SemanticChunker

# --- Preparation & Logging ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


# Constants
MODEL_NAME_CHAT = "groq:llama-3.1-8b-instant"

# --- Data Ingestion ---
text = """
The Great Barrier Reef is the world's largest coral reef system.
Warming ocean temperatures are causing severe bleaching across coral habitats.
Quantum computing utilizes qubits to perform complex calculations at unprecedented speeds.
Tech companies are heavily investing in quantum research to crack advanced cryptography.
Sourdough bread relies on a wild yeast starter for its rise and distinct flavor.
Baking the perfect loaf requires a long fermentation process and high oven heat.
"""
doc = Document(page_content=text)

# --- Chunking with LangChain Native Chunker ---
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
# the standard deviation breakpoint method splits your text whenever the semantic distance between 
# consecutive sentences exceeds
native_chunker = SemanticChunker(
    embedding_model, 
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.25
)
chunks = native_chunker.split_documents([doc])

# --- Vector Store & Retrieval ---
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# --- Prompt & LLM Setup ---
template = """Answer the question thoroughly based on the following context. Do not omit any relevant details.

Context:
{context}

Question: {question}
Answer:"""

# https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate
# https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate/from_template
prompt = PromptTemplate.from_template(template)
llm = init_chat_model(model=MODEL_NAME_CHAT, temperature=0.4)

# --- LCEL Chain ---
# RunnableMap
# A runnable that runs a mapping of runnables in parallel, and returns a mapping of their outputs.
# StrOutputParser
# Extract text content from model outputs as a string.

rag_chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"],  
    })
    | prompt
    | llm
    | StrOutputParser()
)

# --- Execution ---
print("\n--- 2. Running LangChain Native Semantic RAG Query ---")
query_dict = {"question": "Tell me about Quantum computing and research"}
response = rag_chain.invoke(query_dict)
print(f"Result: {response}")