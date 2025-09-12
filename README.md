# PDF Question Answering with Mistral AI

A Streamlit web application that allows you to upload PDF documents and ask questions about their content using Mistral AI and FAISS vector search.

## Features

- 📁 **PDF Upload**: Upload any PDF document
- 🤖 **AI-Powered QA**: Uses Mistral 7B Instruct model via HuggingFace
- 🔍 **Smart Search**: FAISS vector similarity search for relevant context
- 💬 **Chat Interface**: Interactive chat-like experience
- ⚙️ **Configurable**: Adjust model parameters and settings
- 🔒 **Secure**: Local processing, no data storage

## Setup

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Get a HuggingFace API token:
   - Visit https://huggingface.co/settings/tokens
   - Create a new token
   - Set it as environment variable: `export HUGGINGFACEHUB_API_TOKEN=your_token_here`

3. Run the application:
\`\`\`bash
streamlit run app.py
\`\`\`

## Usage

1. **Upload PDF**: Click "Choose a PDF file" and select your document
2. **Configure Settings**: Use the sidebar to adjust model parameters
3. **Ask Questions**: Type your questions in the chat interface
4. **View Sources**: Expand the sources section to see relevant document excerpts

## Models Supported

- Mistral 7B Instruct v0.3 (default)
- Mistral 7B Instruct v0.1

## Requirements

- Python 3.8+
- HuggingFace API token
- GPU recommended for faster processing (CPU also supported)
