# README for Chatbot to Talk to Documents

## Overview
This project creates an interactive chatbot that interacts with documents using a combination of Streamlit for the user interface and Langchain for document processing and conversational AI. The chatbot can handle document uploads, process them into embeddings, and provide conversational retrieval capabilities. It also integrates with Neo4j to visualize and query knowledge graphs.

## Features
- **Simple RAG (Retrieval-Augmented Generation)**: Upload a document, process it with RecursiveCharacterTextSplitter, embed it using FAISS, and interact with it through a conversational retrieval chain.
- **RAG with Neo4j**: Upload a document and generate a knowledge graph using Neo4j. The chatbot can answer questions using a hybrid search approach leveraging the knowledge graph.

## Requirements
- Python 3.8 or higher
- Streamlit
- Langchain
- FAISS
- HuggingFace Transformers
- Py2neo
- NetworkX
- PyVis
- Langsmith
- dotenv

## Installation
1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file in the root directory and add your Langsmith API key:
      ```
      LANGSMITH_API_KEY=<your_langsmith_api_key>
      ```

4. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

## Usage
### Home
- Navigate to the **Home** page to get an overview of the application.

### Simple RAG
1. Go to the **Simple RAG** page.
2. Upload a document (PDF format).
3. The document will be processed, split, and embedded using FAISS.
4. Interact with the document by asking questions in the chat input.

### RAG with Neo4j
1. Go to the **RAG with Neo4j** page.
2. Upload a document (DOCX format).
3. Click on **Load Graph** to generate and visualize the knowledge graph.
4. Interact with the knowledge graph by asking questions in the chat input.

### Neo4j Graph Visualization
1. Provide Neo4j connection details (URI, username, password).
2. Click on **Load Graph** to visualize the knowledge graph.
3. Interact with the graph using the chat input.


## Code Description

### Main Functions

- **streamlit_ui**: Defines the main UI components and navigation.
- **RAG**: Handles document uploading, splitting, embedding, and creating a conversational retrieval chain.
- **RAG_Neo4j**: Manages document uploading, creating a knowledge graph, and chat interaction with the graph.
- **show_graph**: Visualizes the Neo4j graph.
- **get_graph_data**: Queries Neo4j to get graph data.
- **create_networkx_graph**: Creates a NetworkX graph from Neo4j data.
- **visualize_graph**: Visualizes the NetworkX graph using PyVis.

### Utility Functions

- **load_dotenv**: Loads environment variables from the `.env` file.
- **option_menu**: Creates a sidebar navigation menu in Streamlit.
- **st.file_uploader**: Allows users to upload files in the Streamlit interface.
- **st.chat_input**: Handles chat input for interacting with the chatbot.

## Troubleshooting
- Ensure that the correct paths are set for storing temporary files.
- Verify that the Langsmith API key is correctly loaded from the `.env` file.
- Check Neo4j credentials and ensure the Neo4j server is running.


## Acknowledgments
- [Streamlit](https://www.streamlit.io/)
- [Langchain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace](https://huggingface.co/)
- [Neo4j](https://neo4j.com/)
- [NetworkX](https://networkx.org/)
- [PyVis](https://pyvis.readthedocs.io/)

---

This README provides a comprehensive guide to set up, run, and understand the Chatbot to Talk to Documents project. For further information or assistance, refer to the official documentation of the respective libraries used.
