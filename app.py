import os
from dotenv import load_dotenv
import torch
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# -----------------------
# CONFIG
# -----------------------
load_dotenv()

import streamlit as st

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or st.secrets["HUGGINGFACE_TOKEN"] 
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Default parameters
chunk_size = 500
chunk_overlap = 50
num_results = 4
max_tokens = 512
temperature = 0.2

# -----------------------
# STREAMLIT UI ENHANCEMENTS
# -----------------------
st.set_page_config(
    page_title="Ask Your PDF", 
    layout="wide",
    page_icon="📘",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #2E86AB; margin-bottom: 0.5rem;">📘 PDF Question Answering</h1>
    <p style="color: #666; font-size: 1.2rem;">Powered by Mistral AI + FAISS Vector Search</p>
</div>
""", unsafe_allow_html=True)



st.markdown("### 📁 Upload Your PDF Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type="pdf",
    help="Upload a PDF document to ask questions about its content"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        # Save uploaded file locally
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File {uploaded_file.name} uploaded successfully!")

        # -----------------------
        # 1. Load & Split Document
        # -----------------------
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading PDF document...")
        progress_bar.progress(20)
        
        loader = PyPDFLoader(uploaded_file.name)
        documents = loader.load()

      
        progress_bar.progress(40)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        docs = splitter.split_documents(documents)
      

        # -----------------------
        # 2. Build Embeddings
        # -----------------------
       
        progress_bar.progress(60)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # -----------------------
        # 3. Create FAISS index
        # -----------------------
       
        progress_bar.progress(80)
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.vectorstore = vectorstore
        retriever = vectorstore.as_retriever(search_kwargs={"k": num_results})

        progress_bar.progress(100)
        status_text.text("✅ Ready to answer questions!")
        
        progress_bar.empty()
        status_text.empty()

    # -----------------------
    # 4. Hugging Face Client
    # -----------------------
    if HF_TOKEN:
        client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
        
        st.markdown("### 💬 Ask Questions About Your PDF")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve docs
                    retrieved_docs = retriever.get_relevant_documents(prompt)
                    context = "\n\n".join([d.page_content for d in retrieved_docs])

                    # Build prompt
                    system_prompt = f"""
                    Use the following context to answer the question accurately and concisely:
                    {context}

                    Question: {prompt}
                    """

                    try:
                        # Call Mistral on HF Hub
                        response = client.chat_completion(
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                                {"role": "user", "content": system_prompt}
                            ],
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        
                        answer = response.choices[0].message["content"]
                        st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Source {i} (Page {doc.metadata.get('page', '?')})**")
                                st.markdown(f"```\n{doc.page_content[:300]}...\n```")
                                
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.info("Please check your HuggingFace token and try again.")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
    else:
        st.warning("⚠️ Please provide a valid HuggingFace API token in the .env file to start asking questions.")
        st.info("You can get a free token from: https://huggingface.co/settings/tokens")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h3 style="color: #666;">👆 Upload a PDF document to get started</h3>
        <p style="color: #888; margin-top: 1rem;">
            This app uses Mistral AI to answer questions about your PDF documents.<br>
            Your documents are processed locally and securely.
        </p>
    </div>
    """, unsafe_allow_html=True)
  
