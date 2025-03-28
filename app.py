import os
from fastapi import Body, FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq
from pymilvus import MilvusClient
from groq import Groq
from langchain_milvus import Zilliz
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from bson.objectid import ObjectId
import io
from dotenv import load_dotenv

load_dotenv()

app= FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mongo_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongo_uri)
db = client["digibot"]
ud_db = db["user_details"]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

vector_store = None
memory_store = []

def extract_pdf_text(file) -> str:
    text = ""
    pdf_reader = PDFPlumberLoader(file_path=file).load()
    for page in pdf_reader:
        text += page.page_content or ""
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def generate_file_ids(chunks:list[str], file_name:str):
    file_id = [f"{file_name}_{i}" for i in range(len(chunks))]
    return file_id

def create_vector_store(chunks: list[str], document_id: str, file_id: list[str]):
    if document_id is None:
        raise HTTPException(status_code=400, detail="Document ID is required")
    else:
        collection_name = f"id_{document_id}"
        global vector_store
        vector_store = Zilliz.from_texts(
            texts=chunks,
            embedding=embeddings,
            connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token":os.getenv("ZILLIZ_TOKEN")},
            collection_name=collection_name,
            ids=file_id,
            index_params={"index_type": "HNSW", "metric_type": "COSINE"},
            drop_old=False,
        )
        return({"message": "Vector store created successfully."})

def generate_chat_response(document_id: str, bot_name: str, user_input: str):
    vector_store = Zilliz(
        collection_name=f"id_{document_id}",
        connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query=user_input, k=1, score_threshold=0.78)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    if retrieved_docs == []:
        message = [
            {"role": "system", "content": f"""
               You are {bot_name}, a Professional AI assistant. You should respond as a prosessional company representative. The below question/message is not relevant to our company.
                - **Inform the user in a friendly,polite and professional manner that their question/message is unrelated to our Company.** and DO NOT ANSWER THE QUESTION/MESSAGE.
                - Keep it **brief and natural (1 line only).** thats it nothing else.
                - Again saying do not answer this question.
            """},
            {"role": "user", "content": user_input}
        ]
    else:
        context = retrieved_docs[0][0].page_content
        highest_score = retrieved_docs[0][1]
        message = [
            {"role": "system", "content": f"""
            You are {bot_name}, a professional AI assistant for our company.
            
            **Response Guidelines:**
            - Maintain professionalism while keeping responses brief, direct, clear, and helpful.
            - If the user expresses frustration with valid concerns, acknowledge their feelings and provide a constructive answer.
            - If the retrieved context **is irrelevant**, do NOT answer the question from your knowledge base just refrain from answering.
            - If the user is being offensive, politely request respectful communication while staying helpful.
            """},
            {"role": "user", "content": user_input},
            {"role": "system", "content": context}
        ]
    
    response = client.chat.completions.create(
        max_completion_tokens=256,  
        model="llama3-8b-8192",
        messages=message,
        temperature=0.3,
        stream=True
    )

    for chunk in response:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                yield content

@app.get("/")
async def root():
    return {"message": "Connection Successfull"}

@app.post("/chat")
async def chat(data: dict = Body(...)):
    user_input = data.get("user_input")
    document_id = data.get("document_id")
    bot_name = data.get("name")

    return StreamingResponse(generate_chat_response(document_id, bot_name, user_input), media_type="text/plain")
    
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), document_id: str = Form(...)):
    try:
        file_name = file.filename
        user_data = ud_db.find_one({"_id": ObjectId(document_id)})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        if file.filename in user_data["user_files"]:
            pass
        else:
            updated_data = ud_db.find_one_and_update(
                {"_id": ObjectId(document_id)},
                {"$addToSet": {"user_files": file_name}},
            )
            if updated_data:
                content = await file.read()
                pdf_text = extract_pdf_text(io.BytesIO(content))
                text_chunks = get_text_chunks(pdf_text)
                ud_db.find_one_and_update(
                    {"_id": ObjectId(document_id)},
                    {"$push": {"file_ids": len(text_chunks)}}
                )
                file_ids = generate_file_ids(text_chunks, file_name)
                create_vector_store(text_chunks, document_id,file_ids)
            
        return {"message": "Operation completed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.delete("/delete")
async def delete_file(data: dict = Body(...)):
    document_id = data.get("document_id")
    file_name = data.get("file_name")
    try:
        result = ud_db.find_one({"_id": ObjectId(document_id)})
        file_index = result["user_files"].index(file_name)
        file_id = result["file_ids"][file_index]
        unique_id = [f"{file_name}_{i}" for i in range(file_id)]
        vector_store = Zilliz(
            collection_name=f"id_{document_id}",
            connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token":os.getenv("ZILLIZ_TOKEN")},
            embedding_function=embeddings
        )
        vector_store.delete(unique_id)

        # Remove File name
        ud_db.find_one_and_update(
            {"_id": ObjectId(document_id)},
            {"$pull": {"user_files": file_name}},
        )
        # Set file_id to null
        ud_db.find_one_and_update(
            {"_id": ObjectId(document_id)},
            {"$unset": {f"file_ids.{file_index}": 1}},
        )
        # remove null value
        ud_db.find_one_and_update(
            {"_id": ObjectId(document_id)},
            {"$pull": {f"file_ids":None }},
        )
        client = MilvusClient(uri=os.getenv("ZILLIZ_URI_ENDPOINT"), token=os.getenv("ZILLIZ_TOKEN"))
        collection = client.get_collection_stats(f"id_{document_id}")
        if collection["row_count"] == 0:
            client.drop_collection(f"id_{document_id}")
        
        return {"message": "File deleted successfully."} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
