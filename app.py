# import streamlit as st
# from PyPDF2 import PdfReader
# from io import BytesIO
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai

# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function to read the PDF and extract text
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(BytesIO(pdf.read()))
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Function to split the text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to convert text into vector and store it
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embeddings)
#     vector_store.save_local("faiss_index")

# # Function to generate a conversational chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "answer is not available in the context." 
#     Context:\n {context}?\n
#     Question: \n{question}\n   

#     Answer: 
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Function to process user input
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()

#     response = chain({"input_documents": docs, "question": user_question})
    
#     print(response)
#     st.write("Reply:", response["output_text"])

# # Main function to run the Streamlit app
# def main():
#     st.title("ACL Knee Injury Chatbot")
#     st.header("ChatBot built using the Gemini Api")
    
#     # Styling message for ACL focus
#     st.markdown("""
#         <style>
#             .message {
#                 font-size: 18px;
#                 color: #2E8B57;
#                 font-weight: bold;
#             }
#         </style>
#         <div class="message">This chatbot is dedicated exclusively to answering questions about the Anterior Cruciate Ligament (ACL) and related injuries.</div>
#         """, unsafe_allow_html=True)

#     user_question = st.text_input("Ask a question about ACL knee injuries")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload PDF (related to ACL)", type=["pdf"], accept_multiple_files=True)

#         if st.button("Submit and process"):
#             if not pdf_docs:  # Check if no files are uploaded
#                 st.warning("Please upload at least one PDF file related to ACL before processing.")
#             else:
#                 with st.spinner("Processing"):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
#                     if text_chunks:  # Ensure there are chunks to process
#                         get_vector_store(text_chunks)
#                         st.success('Done')
#                     else:
#                         st.warning("No text chunks were created from the PDF. Please check the content.")

# if __name__ == "__main__":
#     main()










#! using the supabase 


# import os
# import faiss
# import numpy as np
# from langchain.schema import Document  # Import Document here
# from io import BytesIO

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import pickle
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Load environment variables
# load_dotenv()

# # Configure Google Generative AI
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if GOOGLE_API_KEY:
#     genai.configure(api_key=GOOGLE_API_KEY)
#     st.write("Google Generative AI configured successfully.")
#     print("Google Generative AI API key configured.")
# else:
#     st.error("Google API key is missing. Please configure it in the environment variables.")

# # FAISS index file for persistence
# FAISS_INDEX_FILE = "faiss_index.bin"
# EMBEDDINGS_FILE = "embeddings.pkl"

# # Function to read the PDF and extract text
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(BytesIO(pdf.read()))
#         for page in pdf_reader.pages:
#             page_text = page.extract_text() or ""
#             text += page_text
#     print(f"Extracted text length: {len(text)} characters.")
#     return text

# # Function to split the text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     print(f"Text split into {len(chunks)} chunks.")
#     return chunks

# # Function to initialize or load the FAISS index
# def initialize_faiss_index(dimension):
#     if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
#         # Load existing FAISS index
#         index = faiss.read_index(FAISS_INDEX_FILE)
#         with open(EMBEDDINGS_FILE, "rb") as f:
#             stored_embeddings = pickle.load(f)
#         print("Loaded existing FAISS index.")
#     else:
#         # Create a new FAISS index
#         index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
#         stored_embeddings = []
#         print("Created a new FAISS index.")
#     return index, stored_embeddings

# # Function to save the FAISS index
# def save_faiss_index(index, stored_embeddings):
#     faiss.write_index(index, FAISS_INDEX_FILE)
#     with open(EMBEDDINGS_FILE, "wb") as f:
#         pickle.dump(stored_embeddings, f)
#     print("FAISS index and embeddings saved.")

# # Function to add documents to FAISS
# # Function to add documents to FAISS
# def add_to_faiss(index, embeddings, stored_embeddings, content):
#     # Reshape the embedding to be a 2D array with shape (1, vector_dimension)
#     embedding_np = np.array(embeddings).astype("float32").reshape(1, -1)
#     index.add(embedding_np)
#     stored_embeddings.append({"embedding": embeddings, "content": content})
#     save_faiss_index(index, stored_embeddings)
#     print("Document added to FAISS index.")


# # Function to query FAISS for similar documents
# def query_faiss(index, stored_embeddings, query_embedding, top_k=5):
#     query_np = np.array([query_embedding]).astype("float32")
#     distances, indices = index.search(query_np, top_k)
#     results = [{"content": stored_embeddings[i]["content"], "distance": distances[0][j]} 
#                for j, i in enumerate(indices[0])]
#     return results

# # Function to generate a conversational chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "Answer is not available in the context." 
#     Context:\n {context}?\n
#     Question: \n{question}\n   
#     Answer: 
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Function to process user input
# def user_input(user_question, index, stored_embeddings):
#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Generate embedding for the query
#     query_embedding = embeddings_model.embed_query(user_question)
#     print("Query embedding generated.")
    
#     # Search FAISS for the most similar documents
#     results = query_faiss(index, stored_embeddings, query_embedding)
#     print(f"Retrieved {len(results)} documents from FAISS.")

#     chain = get_conversational_chain()
    
#     # Prepare Document objects for the chain
#     context_docs = [Document(page_content=result['content'], metadata={}) for result in results]
    
#     response = chain({"input_documents": context_docs, "question": user_question})
    
#     print("Chain response:", response)
#     st.write("Reply:", response["output_text"])

# def main():
#     st.title("ACL Knee Injury Chatbot")
    
#     # Styling message for ACL focus
#     st.markdown("""
#         <style>
#             .message {
#                 font-size: 18px;
#                 color: #2E8B57;
#                 font-weight: bold;
#             }
#         </style>
#         <div class="message">This chatbot is dedicated exclusively to answering questions about the Anterior Cruciate Ligament (ACL) and related injuries.</div>
#         """, unsafe_allow_html=True)

#     user_question = st.text_input("Ask a question about ACL knee injuries")

#     # Initialize FAISS index
#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     dummy_embedding = embeddings_model.embed_documents(["dummy"])[0]  # Get a sample dimension
#     dimension = len(dummy_embedding)
#     index, stored_embeddings = initialize_faiss_index(dimension)

#     if user_question:
#         st.write("Processing your question...")
#         user_input(user_question, index, stored_embeddings)

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload PDF (related to ACL)", type=["pdf"], accept_multiple_files=True)

#         if st.button("Submit and process"):
#             if not pdf_docs:
#                 st.warning("Please upload at least one PDF file related to ACL before processing.")
#             else:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     text_chunks = get_text_chunks(raw_text)
                    
#                     for chunk in text_chunks:
#                         # Generate embeddings and add to FAISS
#                         embedding_1536 = embeddings_model.embed_documents([chunk])[0]
#                         add_to_faiss(index, embedding_1536, stored_embeddings, chunk)
                        
#                     st.success("PDF processed and added to the vector store.")

# if __name__ == "__main__":
#     main()



# !Fast API setup 
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import HTMLResponse, JSONResponse
# # from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# import os
# import faiss
# import numpy as np
# from io import BytesIO
# import pickle
# from langchain.schema import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware

# # Load environment variables
# load_dotenv()

# # Configure Google Generative AI
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if GOOGLE_API_KEY:
#     genai.configure(api_key=GOOGLE_API_KEY)
# else:
#     raise ValueError("Google API key is missing. Please configure it in the environment variables.")

# # FAISS index file for persistence
# FAISS_INDEX_FILE = "faiss_index.bin"
# EMBEDDINGS_FILE = "embeddings.pkl"

# # FastAPI app setup
# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")


# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods, including OPTIONS
#     allow_headers=["*"],  # Allows all headers
# )

# # Function to read the PDF and extract text
# def get_pdf_text(file: UploadFile):
#     text = ""
#     pdf_reader = PdfReader(BytesIO(file.file.read()))
#     for page in pdf_reader.pages:
#         page_text = page.extract_text() or ""
#         text += page_text
#     return text

# # Function to split the text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# # Function to initialize or load the FAISS index
# def initialize_faiss_index(dimension):
#     if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
#         # Load existing FAISS index
#         index = faiss.read_index(FAISS_INDEX_FILE)
#         with open(EMBEDDINGS_FILE, "rb") as f:
#             stored_embeddings = pickle.load(f)
#     else:
#         # Create a new FAISS index
#         index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
#         stored_embeddings = []
#     return index, stored_embeddings

# # Function to save the FAISS index
# def save_faiss_index(index, stored_embeddings):
#     faiss.write_index(index, FAISS_INDEX_FILE)
#     with open(EMBEDDINGS_FILE, "wb") as f:
#         pickle.dump(stored_embeddings, f)

# # Function to add documents to FAISS
# def add_to_faiss(index, embeddings, stored_embeddings, content):
#     # Reshape the embedding to be a 2D array with shape (1, vector_dimension)
#     embedding_np = np.array(embeddings).astype("float32").reshape(1, -1)
#     index.add(embedding_np)
#     stored_embeddings.append({"embedding": embeddings, "content": content})
#     save_faiss_index(index, stored_embeddings)

# # Function to query FAISS for similar documents
# def query_faiss(index, stored_embeddings, query_embedding, top_k=5):
#     query_np = np.array([query_embedding]).astype("float32")
#     distances, indices = index.search(query_np, top_k)
#     results = [{"content": stored_embeddings[i]["content"], "distance": distances[0][j]} 
#                for j, i in enumerate(indices[0])]
#     return results

# # Function to generate a conversational chain
# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "Answer is not available in the context." 
#     Context:\n {context}?\n
#     Question: \n{question}\n   
#     Answer: 
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Route for uploading PDF and processing it
# @app.post("/upload/")
# async def upload_pdf(file: UploadFile = File(...)):
#     text = get_pdf_text(file)
#     text_chunks = get_text_chunks(text)
    
#     # Initialize FAISS index
#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     dummy_embedding = embeddings_model.embed_documents(["dummy"])[0]  # Get a sample dimension
#     dimension = len(dummy_embedding)
#     index, stored_embeddings = initialize_faiss_index(dimension)

#     # Add chunks to FAISS
#     for chunk in text_chunks:
#         embedding_1536 = embeddings_model.embed_documents([chunk])[0]
#         add_to_faiss(index, embedding_1536, stored_embeddings, chunk)
    
#     return {"message": "PDF processed and added to the vector store."}

# # Route for querying based on user question
# @app.post("/query/", response_class=JSONResponse)
# async def query_document(question: str = Form(...)):
#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     index, stored_embeddings = initialize_faiss_index(dimension=1536)
    
#     query_embedding = embeddings_model.embed_query(question)
#     results = query_faiss(index, stored_embeddings, query_embedding)

#     # Generate a conversational chain
#     chain = get_conversational_chain()
#     context_docs = [Document(page_content=result['content'], metadata={}) for result in results]
#     response = chain({"input_documents": context_docs, "question": question})

#     return {"answer": response["output_text"]}

# # Route for rendering the HTML page
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request):
#     return templates.TemplateResponse("index.html", {"request": request})



#! fast api to test in the insomina tool
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import faiss
import numpy as np
from io import BytesIO
import pickle
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("Google API key is missing. Please configure it in the environment variables.")

# FAISS index file for persistence
FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDINGS_FILE = "embeddings.pkl"

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Function to read the PDF and extract text
def get_pdf_text(file: UploadFile):
    text = ""
    pdf_reader = PdfReader(BytesIO(file.file.read()))
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text += page_text
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to initialize or load the FAISS index
def initialize_faiss_index(dimension):
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(EMBEDDINGS_FILE):
        # Load existing FAISS index
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(EMBEDDINGS_FILE, "rb") as f:
            stored_embeddings = pickle.load(f)
    else:
        # Create a new FAISS index
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        stored_embeddings = []
    return index, stored_embeddings

# Function to save the FAISS index
def save_faiss_index(index, stored_embeddings):
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(stored_embeddings, f)

# Function to add documents to FAISS
def add_to_faiss(index, embeddings, stored_embeddings, content):
    # Reshape the embedding to be a 2D array with shape (1, vector_dimension)
    embedding_np = np.array(embeddings).astype("float32").reshape(1, -1)
    index.add(embedding_np)
    stored_embeddings.append({"embedding": embeddings, "content": content})
    save_faiss_index(index, stored_embeddings)

# Function to query FAISS for similar documents
def query_faiss(index, stored_embeddings, query_embedding, top_k=5):
    query_np = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_np, top_k)
    results = [{"content": stored_embeddings[i]["content"], "distance": distances[0][j]} 
               for j, i in enumerate(indices[0])]
    return results

# Function to generate a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "Answer is not available in the context." 
    Context:\n {context}?\n
    Question: \n{question}\n   
    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Endpoint for uploading PDF and processing it
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    text = get_pdf_text(file)
    text_chunks = get_text_chunks(text)
    
    # Initialize FAISS index
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    dummy_embedding = embeddings_model.embed_documents(["dummy"])[0]  # Get a sample dimension
    dimension = len(dummy_embedding)
    index, stored_embeddings = initialize_faiss_index(dimension)

    # Add chunks to FAISS
    for chunk in text_chunks:
        embedding_1536 = embeddings_model.embed_documents([chunk])[0]
        add_to_faiss(index, embedding_1536, stored_embeddings, chunk)
    
    return {"message": "PDF processed and added to the vector store."}

# Endpoint for querying based on user question
@app.post("/query/")
async def query_document(question: str = Form(...)):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index, stored_embeddings = initialize_faiss_index(dimension=1536)
    
    query_embedding = embeddings_model.embed_query(question)
    results = query_faiss(index, stored_embeddings, query_embedding)

    # Generate a conversational chain
    chain = get_conversational_chain()
    context_docs = [Document(page_content=result['content'], metadata={}) for result in results]
    response = chain({"input_documents": context_docs, "question": question})

    return {"answer": response["output_text"]}

# Health check endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the ACL Knee Injury Chatbot API"}
