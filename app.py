import os
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_milvus import Zilliz
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), temperature=0, model="llama-3.3-70b-versatile")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

async def generate_chat_response(document_id: str, bot_name: str, user_input: str):
    # Search for relevant document context
    vector_store = Zilliz(
        collection_name=f"id_{document_id}",
        connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )
    
    # Retrieve top documents with a relevance score threshold
    retrieved_docs = vector_store.similarity_search_with_relevance_scores(query=user_input, k=1, score_threshold=0.75)
    
    if retrieved_docs == []:
        message = [HumanMessage(content=f"""
               You are {bot_name}, a Professional AI assistant. You should respond as a prosessional company representative. The below question/message is not relevant to our company.
                - **Inform the user in a friendly,polite and professional manner that their question/message is unrelated to our Company.** and DO NOT ANSWER THE QUESTION/MESSAGE.
                - Keep it **brief and natural (1 line only).** thats it nothing else.
                - Again saying do not answer this question.
            """),HumanMessage(content=user_input)]
    else:
        context = retrieved_docs[0][0].page_content
        
        message = [
            HumanMessage(content=f"""
            You are {bot_name}, a professional AI assistant for our company.
            Give responses as if you are a member of the company or a representative of the company.
            
            **Response Guidelines:**
            - Responses should not exceed 225 words, with proper format.
            - Only answer if relevant context from the document exists.
            - If no relevant context is found, politely steer the conversation back to document-related topics.
            - Do not provide general knowledge or answer off-topic questions.- If the retrieved context **is irrelevant**, do NOT answer the question from your knowledge base just refrain from answering.
            - If the user is being offensive, politely request respectful communication while staying helpful.
            - **Maintain all Markdown syntax exactly as provided, including links, bold text, and formatting. Do not alter or reformat them.**
            
            Context: {context}
            """),
            HumanMessage(content=user_input),
        ]
    
    response = llm.stream(message)
    for chunks in response:
        yield chunks.content

@app.get("/")
async def root():
    return {"message": "Connection Successfull"}

@app.post("/chat")
async def chat(data: dict = Body(...)):
    document_id = data.get("document_id")
    bot_name = data.get("name")
    user_input = data.get("user_input")

    return StreamingResponse(generate_chat_response(document_id, bot_name, user_input), media_type="text/plain")