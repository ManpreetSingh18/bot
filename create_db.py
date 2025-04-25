from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Directory where PDFs are stored
PDF_FOLDER = "data/"

# Step 1: Read PDF documents from directory
def fetch_pdfs_from_folder(folder_path):
    pdf_loader = DirectoryLoader(folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    loaded_docs = pdf_loader.load()
    return loaded_docs

# Load and print total pages
all_docs = fetch_pdfs_from_folder(PDF_FOLDER)
print("📄 Total PDF pages loaded:", len(all_docs))

# Step 2: Break long texts into smaller pieces
def split_into_chunks(documents, size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

chunks = split_into_chunks(all_docs)
print("✂️ Total text chunks created:", len(chunks))

# Step 3: Load HuggingFace embedding model
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedding_model()

# Step 4: Generate and save vector embeddings using FAISS
VEC_STORE_PATH = "vectorstore/db_faiss"
vector_index = FAISS.from_documents(chunks, embedder)
vector_index.save_local(VEC_STORE_PATH)
print(f"✅ Vector store saved at: {VEC_STORE_PATH}")
