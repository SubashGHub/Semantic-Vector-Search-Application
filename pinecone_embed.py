from pinecone import Pinecone, ServerlessSpec
import os
import tiktoken
from uuid import uuid4
import re

# Connect to Pinecone
try:
    pc = Pinecone(api_key=os.environ['Pinecone_API'])  # Replace with your Pinecone API key
    print('Pinecone connection done.')
except Exception as e:
    print('Pinecone connection error.')
    print(e)


class PineconeOperations:

    # Create Index
    def create_index(self, index_name):
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1024,  # E5-large has 1024-dimensional embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f'Index "{index_name}" is created.')
        else:
            print(f'Index "{index_name}" already exists.')

    # Chunk the text
    def chunk_text(self, text, max_tokens=110, overlap=20):
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk = tokenizer.decode(tokens[start:end])
            chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    def qa_chunks(self, text):
        # Split by Q: and preserve the question-answer blocks
        qa_blocks = re.split(r'\nQ: ', text)
        qa_chunks = [block.strip() for block in qa_blocks if block.strip()]
        return qa_chunks

    # Step 4: Embed Text
    def create_vector_embed(self, text_list):
        raw_embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=text_list,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        embeddings = [e["values"] for e in raw_embeddings]
        print('Embed created.')
        return embeddings

    # Step 5: Upsert to Pinecone
    def upsert_embeddings(self, index_name, texts, vectors):
        items = []
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        for text, vector in zip(texts, vectors):
            metadata = {"text": text}
            items.append({
                "id": str(uuid4()),
                "values": vector,
                "metadata": metadata
            })
        index.upsert(items)
        print(f'Upserted {len(items)} vectors to Pinecone index.')

    # Step 6: Query from Pinecone
    def query_index(self, index_name, query_text, top_k=1):
        index = pc.Index(index_name)
        embed = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={"input_type": "query", "truncate": "END"}
        )[0]
        results = index.query(vector=embed['values'], top_k=top_k, include_metadata=True)
        for match in results['matches']:
            raw_score = match['score'] * 100
            score = f"Match Score: {match['score']:.2%}"
            match = f"Q&A: {match['metadata']['text']}"
            if int(raw_score) > 80:
                return score, match
            else:
                return f"""
                {score} is below 80%.
                Ask Questions from the given list. 
                """

