from pinecone_embed import PineconeOperations
from datetime import datetime

db = PineconeOperations()

index_name = "java-q-a"

# Create index if not exists
db.create_index(index_name)
# Read and chunk the file
with open("java_Q&A.txt", "r", encoding="utf-8") as f:
    text = f.read()

# chunks = chunk_text(text)
chunks = db.qa_chunks(text)
print(f"Total chunks created: {len(chunks)}")

# Save chunks (optional logging)
with open("java_chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"{datetime.now()} --- Chunk {i + 1} ---\n")
        f.write(chunk + "\n\n")

# Create vector embeddings
embeddings = db.create_vector_embed(chunks)

# Upsert into Pinecone
db.upsert_embeddings(index_name, chunks, embeddings)
