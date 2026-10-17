"""
10a_ReRanking.py: Initializes the re-ranker LLM with default settings. The prompt template casually asks the LLM to provide a comma-separated list of ranked indices (e.g., 2,1,3,0,...). It lacks formatting enforcement, leaving it vulnerable to LLM conversational fluff or explanations.
10b_ReRanking.py: Enforces strict adherence by explicitly setting the model's temperature=0.0. The prompt template is re-engineered with severe structural guardrails: it instructs the LLM to output a JSON-like array wrapper ([index, index, index]) and explicitly bans conversational intros, summaries, or explanatory tokens.

10a_ReRanking.py splits the string directly by commas:
    indices = [int(x.strip()) - 1 for x in ranking_response.split(",") if x.strip().isdigit()]  
10b_ReRanking.py adds an intermediate text-cleaning layer to strip away the brackets before processing the comma breakdown:
    cleaned_response = ranking_response.replace("[", "").replace("]", "")
    indices = [int(x.strip()) - 1 for x in cleaned_response.split(",") if x.strip().isdigit()]

10a_ReRanking.py is written as a simple, procedural top-to-bottom script execution loop. It runs exactly one hardcoded evaluation question against the pipeline during execution.
10b_ReRanking.py abstracts the entire two-stage RAG mechanism into a reusable python function framework: run_reranking_pipeline(test_title, query_text). This allows it to easily execute an automated test suite under an if __name__ == "__main__": block, sequentially evaluating three completely different technical multi-hop cybersecurity scenarios.
"""
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
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

# ==============================================================================
# 1. INITIAL PREPARATION: Environment & Logging
# ==============================================================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ==============================================================================
# 2. DATA INGESTION & PROCESSING
# ==============================================================================
FILE_NAME = "cybersecurity_data/cybersecurity_concepts.txt"
try:
    loader = TextLoader(FILE_NAME)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
except Exception as e:
    print(f"Error loading file: {e}. Please ensure {FILE_NAME} exists.")
    docs = []

# ==============================================================================
# 3. RETRIEVAL & COLD STORAGE ARCHITECTURE SETUP
# ==============================================================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)
vectorstore = FAISS.from_documents(docs, embeddings)

# Retrieve a larger candidate window (k=8) to ensure high recall
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ==============================================================================
# 4. STAGE 2: RE-RANKER CONFIGURATION (LLM Judge)
# ==============================================================================
MODEL_NAME_CHAT = "groq:llama-3.1-8b-instant"
llm = init_chat_model(MODEL_NAME_CHAT, temperature=0.0) # Low temperature for strict adherence

ranking_template = """You are a precise ranking engine. Your job is to re-order the given documents based on how well they answer the user's question.

User Question: "{question}"

Documents to evaluate:
{documents}

TASK:
Identify the indices of the documents that contain the actual answer to the question. List them in order of most relevant to least relevant.

CRITICAL FORMAT RULES:
- Output your response exactly like this template: [index, index, index]
- Example of correct output: [3, 1, 4]
- Do not include the word "Result" or any extra text.
- Do not list all numbers if they are not relevant.
- Output ONLY the bracketed list.

Your JSON-like list response:"""

ranking_prompt = PromptTemplate.from_template(ranking_template)
ranking_chain = ranking_prompt | llm | StrOutputParser()

# ==============================================================================
# 5. EXECUTION PIPELINE FUNCTION
# ==============================================================================
def run_reranking_pipeline(test_title: str, query_text: str):
    print("\n" + "="*70)
    print(f"🔥 {test_title}")
    print("="*70)
    print(f"🔍 User Query: '{query_text}'\n")

    # --- Stage 1: Broad Vector Search ---
    retrieved_docs = retriever.invoke(query_text)
    
    print("⏳ Stage 1 Vector Search Results (Original Order sent to Re-ranker):")
    for idx, doc in enumerate(retrieved_docs):
        print(f"  [{idx + 1}] {doc.page_content[:110].strip()}...")

    # --- Stage 2: LLM Precision Re-ranking ---
    doc_lines = [f"{i+1}. {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
    formatted_docs_str = "\n".join(doc_lines)

    ranking_response = ranking_chain.invoke({
        "question": query_text, 
        "documents": formatted_docs_str
    }).strip()

    # print(f"\n🧠 Raw Re-ranker Array Output: {ranking_response}")

    # # Parse indices safely
    # try:
    #     # Convert 1-based UI indices back to 0-based Python list indexes
    #     indices = [int(x.strip()) - 1 for x in ranking_response.split(",") if x.strip().isdigit()]
    #     reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
    # except Exception as e:
    #     print(f"❌ Parsing failed: {e}. Falling back to baseline.")
    #     reranked_docs = retrieved_docs

# --- Stage 2: LLM Precision Re-ranking ---
    ...
    print(f"\n🧠 Raw Re-ranker Array Output: {ranking_response}")

    # FIX: Strip brackets before splitting
    cleaned_response = ranking_response.replace("[", "").replace("]", "")

    # Parse indices safely
    try:
        # Convert 1-based UI indices back to 0-based Python list indexes
        indices = [int(x.strip()) - 1 for x in cleaned_response.split(",") if x.strip().isdigit()]
        reranked_docs = [retrieved_docs[i] for i in indices if 0 <= i < len(retrieved_docs)]
    except Exception as e:
        print(f"❌ Parsing failed: {e}. Falling back to baseline.")
        reranked_docs = retrieved_docs
        
    # --- Final Presentation ---
    print("\n📊 Final Top 3 Grounded Results Passed to LLM Context:")
    for rank, doc in enumerate(reranked_docs[:3], 1):
        print(f"  🏆 Rank {rank}: {doc.page_content.strip()}")

# ==============================================================================
# 6. RUNNING THE SAMPLES
# ==============================================================================
if __name__ == "__main__":
    # Sample 1: Evaluates identity-first frameworks, administrative control, and device tracking.
    # Expects the retriever to rank Zero Trust, IAM/PAM, and EDR/XDR details at the top.
    query1 = "What is the modern approach to enterprise security, and how do we protect privileged access and endpoints?"
    run_reranking_pipeline("SAMPLE TEST 1: Identity Frameworks & Endpoint Security", query1)

    # Sample 2: Evaluates architectural shifts against perimeter models and cloud posture stability.
    # The Re-ranker must prioritize structural network design and CSPM self-healing over software pipelines.
    # query2 = "How does Zero Trust Architecture differ from traditional security, and how does CSPM help manage cloud risks?"
    # run_reranking_pipeline("SAMPLE TEST 2: Network Architecture vs Cloud Misconfigurations", query2)

    # Sample 3: Focuses heavily on CI/CD pipeline scanning and real-time behavioral monitoring.
    # Expects the system to elevate DevSecOps "Shift-Left" concepts and automated threat responses.
    # query3 = "How can we use automation to detect software vulnerabilities early and monitor endpoint threats in real time?"
    # run_reranking_pipeline("SAMPLE TEST 3: DevSecOps Automation & Threat Detection", query3)