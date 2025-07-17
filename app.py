import os
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_milvus import Zilliz
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources once
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    model="llama-3-3-70b-versatile"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Reusable vector store generator
def get_vector_store(document_id: str):
    return Zilliz(
        collection_name=f"id_{document_id}",
        connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )

# Streaming response generator
async def generate_chat_response(document_id: str, bot_name: str, user_input: str):
    vector_store = get_vector_store(document_id)
    retrieved_docs = vector_store.similarity_search(
        query=user_input, k=3
    )

    context_blocks = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(context_blocks)

    message = [
        HumanMessage(content=f"""
                     You are {bot_name}, a professional AI assistant representing our company.
                     
                     Use only the context provided below to answer. If the context is unrelated to the question, you must **politely inform the user that the question is unrelated and you cannot help with it. Do not answer from your own knowledge base**.
                     
                     **Guidelines:**
                     - Max response length: 225 words.
                     - Maintain markdown format as-is, including links, bold, etc dont include anything extra of your own.
                     - If the context is insufficient or off-topic, do not attempt to improvise.
                     - If user input is offensive, respond professionally and ask for respectful interaction.
                     
                     Context:
                     {context}
                     
                     Question:
                     {user_input}
""")]

    response = llm.stream(message)
    for chunk in response:
        yield chunk.content

@app.get("/")
async def root():
    return JSONResponse(content="Connection Successful")

@app.post("/chat")
async def chat(data: dict = Body(...)):
    document_id = data.get("document_id")
    bot_name = data.get("name")
    user_input = data.get("user_input")

    return StreamingResponse(
        generate_chat_response(document_id, bot_name, user_input),
        media_type="text/plain"
    )