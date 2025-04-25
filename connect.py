import os
import warnings

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Ignore future warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# === Step 1: Load HuggingFace Model ===
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"

def initialize_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HUGGINGFACE_TOKEN,
            "max_length": "512"
        },
        task="text-generation"
    )

# === Step 2: Define Custom Prompt Format ===
PROMPT_TEMPLATE = """
Refer strictly to the context given to answer the user's query.
If the answer isn't known, say "I don't know." Avoid guessing or fabricating.
Don't include unrelated content.

Context: {context}
Question: {question}

Reply directly without extra introduction.
"""

def create_prompt(template_text):
    return PromptTemplate(
        template=template_text,
        input_variables=["context", "question"]
    )

# === Step 3: Load Vector Index from FAISS ===
VEC_STORE_DIR = "vectorstore/db_faiss"
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc_search_db = FAISS.load_local(
    VEC_STORE_DIR,
    embedding_function,
    allow_dangerous_deserialization=True
)

# === Step 4: Build Retrieval-Based QA Pipeline ===
qa_pipeline = RetrievalQA.from_chain_type(
    llm=initialize_llm(MODEL_REPO),
    chain_type="stuff",
    retriever=doc_search_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": create_prompt(PROMPT_TEMPLATE)}
)

# === Step 5: Accept Query & Generate Response ===
query = input("🔍 Ask your question: ")
output = qa_pipeline.invoke({'query': query})

# Display response
print("\n🧠 Answer:", output["result"])
# To show source documents, uncomment below:
# print("📚 Sources:", output["source_documents"])
