# 1. Ingest PDF File
# 2. Extract text from PDF files and split into small chunks
# 3. Send the chunks to the embedding Model 
# 4. Save embeddings to a vector database 
# 5. Perform similarity search in vector db 
# 6. Retrieve the similar documents and present them to the user

from langchain_community.document_loaders import PyPDFLoader

doc_path = "./data/BOI.pdf"
model = "gemma:2b"

if doc_path:
    loader = PyPDFLoader(doc_path)
    data = loader.load()
    print("✅ Done loading....")
else:
    print("⛔ Upload a PDF file")

# Print first 100 characters
content = data[0].page_content
print(content[:100])

# ==== Extract Text from PDF Files and Split into Small Chunks ====

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print("done splitting....")

# ===== Add to vector database ===
import ollama

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database....")

## === Retrieval ===
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough

# set up our model to use
llm = ChatOllama(model=model, streaming=False)

_ = llm.invoke("say hi")  # warm-up

retriever = vector_db.as_retriever()
# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nWelcome to the BOI Document Chatbot! Type 'quit' to exit.\n")

while True:
    user_question = input("Ask a question about the BOI document: ")
    if user_question.lower() == 'quit':
        print("\nThank you for using the BOI Document Chatbot!")
        break
    
    try:
        response = chain.invoke(user_question)
        print("\nAnswer:", response, "\n")
    except Exception as e:
        print("\nError:", str(e), "\n")