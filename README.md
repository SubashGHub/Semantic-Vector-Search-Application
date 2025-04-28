
# Semantic Vector Search Application



                                                                                                                                
## 📌 Overview
- This project is a semantic vector search application built using **Python, Streamlit, and Pinecone.**
- It demonstrates how semantic search works by retrieving relevant answers based on the meaning of the query rather than relying on exact keyword matches.

- Unlike traditional keyword search, where the query and document must share the same words, this application uses vector embeddings to understand the context and intent behind a query. Even when different words or phrasing are used, the system successfully retrieves the most relevant matches by comparing semantic similarity.

- The primary goal of this project is to showcase the power of vector databases for intelligent information retrieval.
- It offers a simple and interactive example of how embeddings, chunking, and similarity search can be combined to build smarter search systems.
## ✨ Features
- **Data Processing:**

    - Reads text data (e.g., Java Q&A pairs).

    - Chunks content intelligently using tiktoken.

    - Generates vector embeddings using Pinecone's Inference API (multilingual-e5-large model).

- **Vector Storage:**

    - Creates a serverless Pinecone index.

    - Upserts processed text chunks along with their vector embeddings.

- **Semantic Search:**

    - Takes user queries, generates embeddings, and performs similarity search against the Pinecone index.

    - Returns the most relevant text chunks based on semantic meaning.

- **Web Interface:**

    - Streamlit-based simple, interactive UI.

    - Users can input questions and view ranked results.

    - Includes session-based query limiting and reset functionality.
## ⚙️ Setup and Usage

1. Clone the Repository


```
 git clone https://github.com/SubashGHub/Semantic-Vector-Search-Application.git
```
2. Install Required Libraries

```
pip install -r requirements.txt
```
3. Set Up Environment Variables

    - Create Account in Pinecone database to get API Key 
    - Set up your Pinecone API into system "Environmental Variable".

4. Populate the Pinecone Index

Run the following to process the dataset and upload embeddings:

```
python main.py
```
5. Launch the Web Application

```
streamlit run app_UI.py
```
## 🔥 Example

- Enter a Java-related question in the UI, such as:

    **"Why do we need constructor in Java?"**
    
    **"Explain the concept of Inheritance?"**

- The app will retrieve the most semantically relevant answers from the dataset.
## 🛠 Tech Stack


**Backend:** Python

**Vector Database:** Pinecone (pinecone-client, Inference API)

**Embeddings Model:** multilingual-e5-large

**Text Tokenization:** tiktoken

**Web UI:** Streamlit

**Other Libraries:** os, re, uuid, datetime, time