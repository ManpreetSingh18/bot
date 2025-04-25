import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Directory where FAISS vector store is stored
VECTORS_PATH = "vectorstore/db_faiss"

# Cache the vector store loading process for better performance
@st.cache_resource
def load_vector_database():
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = FAISS.load_local(VECTORS_PATH, embed_model, allow_dangerous_deserialization=True)
    return vector_db

# Create a LangChain-style prompt using the provided format
def build_prompt_template(template_str):
    return PromptTemplate(template=template_str, input_variables=["context", "question"])

# Load the language model from HuggingFace Inference API
def initialize_llm(repo_id, api_token):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_new_tokens=512,
        task="text-generation",
        huggingfacehub_api_token=api_token
    )

def run_chatbot():
    st.title("💬 Ask Me Anything - Chatbot")

    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display previous messages
    for entry in st.session_state.history:
        st.chat_message(entry['role']).markdown(entry['content'])

    user_input = st.chat_input("Enter your question...")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.history.append({'role': 'user', 'content': user_input})

        # Template used to instruct the LLM
        SYSTEM_TEMPLATE = """
        Use the provided context to respond to the user's question.
        If the answer isn't available in the context, say you don't know.
        Don't fabricate information or give extra commentary.

        Context: {context}
        Question: {question}

        Answer the question concisely below:
        """

        HUGGINGFACE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HUGGINGFACE_API_KEY = os.environ.get("HF_TOKEN")

        try:
            vector_store = load_vector_database()
            if vector_store is None:
                st.error("Unable to load the vector store.")
                return

            qa_pipeline = RetrievalQA.from_chain_type(
                llm=initialize_llm(repo_id=HUGGINGFACE_MODEL_ID, api_token=HUGGINGFACE_API_KEY),
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": build_prompt_template(SYSTEM_TEMPLATE)}
            )

            response = qa_pipeline.invoke({"query": user_input})

            final_answer = response.get("result", "No answer generated.")
            sources = response.get("source_documents", [])

            answer_output = f"### ✅ Response:\n\n{final_answer}\n\n"
            sources_output = "### 📚 Source References:\n\n"

            for doc in sources:
                title = doc.metadata.get("title", "Untitled")
                source = doc.metadata.get("source", "Unknown Source")
                page = doc.metadata.get("page_label", "N/A")
                preview = doc.page_content[:300]

                sources_output += f"**Title:** {title}  \n**File:** {source}  \n**Page:** {page}  \n**Excerpt:** {preview}...\n\n"

            st.chat_message("assistant").markdown(answer_output + sources_output)
            st.session_state.history.append({'role': 'assistant', 'content': answer_output + sources_output})

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")

if __name__ == "__main__":
    run_chatbot()
