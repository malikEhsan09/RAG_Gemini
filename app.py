import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to read the PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to convert text into vector and store it
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# Function to generate a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "answer is not available in the context." 
    Context:\n {context}?\n
    Question: \n{question}\n   

    Answer: 
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question})
    
    print(response)
    st.write("Reply:", response["output_text"])

# Main function to run the Streamlit app
def main():
    st.title("ACL Knee Injury Chatbot")
    st.header("ChatBot built using the Gemini Api")
    
    # Styling message for ACL focus
    st.markdown("""
        <style>
            .message {
                font-size: 18px;
                color: #2E8B57;
                font-weight: bold;
            }
        </style>
        <div class="message">This chatbot is dedicated exclusively to answering questions about the Anterior Cruciate Ligament (ACL) and related injuries.</div>
        """, unsafe_allow_html=True)

    user_question = st.text_input("Ask a question about ACL knee injuries")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF (related to ACL)", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit and process"):
            if not pdf_docs:  # Check if no files are uploaded
                st.warning("Please upload at least one PDF file related to ACL before processing.")
            else:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:  # Ensure there are chunks to process
                        get_vector_store(text_chunks)
                        st.success('Done')
                    else:
                        st.warning("No text chunks were created from the PDF. Please check the content.")

if __name__ == "__main__":
    main()










#! using the supabase 


# import streamlit as st
# from PyPDF2 import PdfReader
# from io import BytesIO
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from supabase import create_client, Client

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai

# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize Supabase client
# SUPABASE_URL = os.getenv("SUPABASE_URL")
# SUPABASE_ANON_KEY = os.getenv("SUPABASE_API_KEY")
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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

# # Function to upload document and embedding to Supabase
# def upload_to_supabase(content, embedding_768=None, embedding_1536=None):
#     data = {
#         "content": content,
#     }
#     if embedding_768 is not None:
#         data["embedding_768"] = embedding_768
#     if embedding_1536 is not None:
#         data["embedding_1536"] = embedding_1536
    
#     response = supabase.from_("documents").insert(data).execute()
    
#     if response.error:
#         st.error(f"Error uploading to Supabase: {response.error.message}")
#     else:
#         st.success("Document uploaded successfully!")

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
#     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Retrieve documents from Supabase based on user question (similarity search)
#     query_embedding = embeddings_model.embed_query(user_question)  # Correct method for query embedding
    
#     response = supabase.rpc('match_documents', {
#         'query_embedding': query_embedding,
#         'match_threshold': 0.8,
#         'match_count': 5,
#     }).execute()

#     if response.error:
#         st.error(f"Error fetching results: {response.error.message}")
#         return
    
#     docs = response.data
    
#     chain = get_conversational_chain()
    
#     # Prepare documents for the chain (assuming docs contain content)
#     context_docs = [{"context": doc['content']} for doc in docs]
    
#     response = chain({"input_documents": context_docs, "question": user_question})
    
#     print(response)
#     st.write("Reply:", response["output_text"])

# # Main function to run the Streamlit app
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
#                     text_chunks = get_text_chunks(raw_text)  # Define text_chunks here
                    
#                     embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#                     for chunk in text_chunks:
#                         # Generate both types of embeddings if needed; ensure you use correct models.
#                         embedding_768 = None  # Replace this with actual call if using a model that outputs this dimension.
#                         embedding_1536 = embeddings_model.embed_documents([chunk])[0]  # Assuming this outputs a 1536-dim vector.
                        
#                         upload_to_supabase(chunk, embedding_768=embedding_768, embedding_1536=embedding_1536)

# if __name__ == "__main__":
#     main()






# * Chroma database