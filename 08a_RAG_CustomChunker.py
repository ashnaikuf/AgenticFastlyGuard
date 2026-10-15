"""
================================================================================
This script introduces an advanced document pre-processing strategy: **Custom Semantic Chunking**. 
Instead of relying on rigid token or character counts 
which can slice paragraphs mid-thought, this script builds a mathematical chunker 
that evaluates sentence-by-sentence cosine similarity using a local HuggingFace 
`SentenceTransformer` model to group conceptually aligned text.

1. CUSTOM CLASS DEFINITION (`ThresholdSemanticChunker`): 
   - Splits a large corpus into distinct individual sentences.
   - Generates vector representations using `all-MiniLM-L6-v2`.
   - Iterates line-by-line, analyzing the cosine similarity between sentence vectors.
   - Joins sentences together if they cross an operational threshold; otherwise, 
     it breaks the text flow to spawn a brand new semantic chunk.
2. CHUNKING COMPILATION: Runs a sample document containing non-linear topics 
   (reefs, quantum computing, sourdough baking) through the customized chunker.
3. RAG RUNTIME ORCHESTRATION: Compiles a standard FAISS index with OpenAI embeddings 
   and links it to a Groq Llama-3.1 model via an LCEL pipeline map to address 
   complex domain-specific questions.
================================================================================
"""

import os
import logging
import warnings
from dotenv import load_dotenv

# LangChain Core & Models
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

# Vector Store
from langchain_community.vectorstores import FAISS

# Semantic Analysis Tools
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Preparation & Logging ---
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Constants
# uv add sentence-transformers
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
MODEL_NAME_SENTENCE_TRANSFORMER = "all-MiniLM-L6-v2" 
MODEL_NAME_CHAT = "groq:llama-3.1-8b-instant"
MODEL_EMBEDDING= "text-embedding-3-small"

# --- Custom Threshold Semantic Chunker ---
class ThresholdSemanticChunker:
    """
    A custom document splitter that uses SentenceTransformer embeddings 
    to group sentences based on a cosine similarity threshold.
    """
    def __init__(self, model_name=MODEL_NAME_SENTENCE_TRANSFORMER, threshold=0.7):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold 

    def _split(self, text: str):
        # --- Step 1: Split raw text into individual sentences ---
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        if not sentences: return []
        
        # --- Step 2: Generate Vector Embeddings ---
        embeddings = self.model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]

        # --- Debug Step: Show line-by-line semantic similarities ---
        print("\n--- Line-by-Line Semantic Similarity Analysis ---")
        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            print(f"Similarity between Line {i} and Line {i+1}: {sim:.4f} (Threshold: {self.threshold})")
            
            # --- Step 3: Iterative Semantic Grouping ---
            if sim >= self.threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        chunks.append(" ".join(current_chunk))
        return chunks
    
    def split_documents(self, docs):
        result = []
        for doc in docs:
            for chunk in self._split(doc.page_content):
                result.append(Document(page_content=chunk, metadata=doc.metadata))
        return result

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

# --- Chunking with Custom Chunker ---
# threshold: higher = smaller/more specific chunks; lower = larger/looser chunks
THRESHOLD = 0.30 
chunker = ThresholdSemanticChunker(threshold=THRESHOLD)
chunks = chunker.split_documents([doc])


# ==============================================================================
# 4. OUTPUT CHUNKING RESULTS (Before Vector Store / LLM RAG)
# ==============================================================================
print("\n📌 Semantic Chunks:")
for idx, chunk in enumerate(chunks):
    print(f"\nChunk {idx+1}:")
    print(f"---")
    print(f"{chunk.page_content}") # Accessing page_content since split_documents returns wrapped Document objects


# --- Vector Store & Retrieval ---
print("\nInitializing FAISS Vector Store...")
embedding_model = OpenAIEmbeddings(model=MODEL_EMBEDDING, dimensions=512)
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Prompt & LLM Setup ---
template = """Answer the question completely based on the following context. 
Provide all relevant details found in the context.

Context:
{context}

Question: {question}
Answer:"""
# A prompt template consists of a string template. It accepts a set of parameters from the user that can be used to generate a prompt for a language model.
# from_template - Load a prompt template from a template.
prompt = PromptTemplate.from_template(template)
# Initialize a chat model from any supported provider using a unified interface.
llm = init_chat_model(model=MODEL_NAME_CHAT, temperature=0.4)

# The reason lambda is used in this specific chain boils down to one key difference: 
# the shape of the input data being passed into the chain.
# script passes a dictionary ({"question": "..."}), 
# the input flowing into the start of the chain is x = {"question": "..."}.

# --- LCEL Chain ---
# https://reference.langchain.com/javascript/langchain-core/runnables/RunnableMap
# https://reference.langchain.com/python/langchain-core/output_parsers/string/StrOutputParser
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
print("\n--- Running Custom Semantic RAG Query ---")
query_dict = {"question": "Tell me about Quantum computing and research"}
response = rag_chain.invoke(query_dict)
print(f"\tResult: {response}")