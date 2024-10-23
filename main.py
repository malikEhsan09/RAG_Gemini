import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import SupabaseVectorStore
from langchain_community.vectorstores import SupabaseVectorStore
# from langchain.llms.openai import OpenAI
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI()

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Supabase Client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Initialize Vector Store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
)

# Initialize OpenAI LLM
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))


def read_pdf(file: UploadFile):
    pdf = PdfReader(file.file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    text_length = len(text)
    for i in range(0, text_length, chunk_size - overlap):
        end_index = i + chunk_size
        if end_index > text_length:
            end_index = text_length
        chunks.append(text[i:end_index])
    return chunks


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "Only PDF files are accepted."}, status_code=400)

    text = read_pdf(file)
    chunks = chunk_text(text)

    # Add texts to the vector store
    vector_store.add_texts(chunks)

    return {"message": "Document uploaded and processed successfully."}


@app.post("/query")
async def query_document(query: str):
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    response = qa_chain.run(query)
    return {"response": response}
