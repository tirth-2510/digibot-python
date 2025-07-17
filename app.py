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
    model="llama-3.3-70b-versatile"
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
def generate_chat_response(document_id: str, bot_name: str, user_input: str):
    vector_store = get_vector_store(document_id)
    retrieved_docs = vector_store.similarity_search(
        query=user_input, k=3
    )
    
    context_blocks = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(context_blocks)

    message = [
        HumanMessage(content=f"""
You are {bot_name}, a professional AI assistant representing our company.
Your task is to answer user queries using **only** the context provided below. You are not allowed to use your own knowledge, assumptions, or common sense reasoning beyond the supplied context.

---

**Behavioral Guidelines:**

1. **Strict Context Adherence**:  
   - You must rely **exclusively** on the given context.  
   - If the context lacks enough information or is unrelated, you must politely inform the user that the question is unrelated and you cannot help with it dont over explain
   example: I am sorry, but the question seems irrelevant and I cannot help you with it, please try rephrasing it?

2. **No Hallucination or Guessing**:  
   - Do not fabricate details or expand with general knowledge.  
   - Never attempt to “fill in the gaps.” Always defer when context is incomplete.

3. **Formatting Rules (CRITICAL)**:
   - **DO NOT** introduce new formatting such as bold, italic, markdown links, headers, or bullet points unless they are already present in the context.  
   - If the context includes formatting (e.g. `**bold**`, `*italic*`, `[links](url)`), **copy it exactly** as shown.  
   - Do not stylize or enhance the response in any way beyond this.

4. **Answer Clarity and Precision**:  
   - Be concise but accurate. Extract key details exactly, without paraphrasing when possible.  
   - Use bullet points or clear separation if multiple items are referenced in the context.

5. **Response Length Constraint**:  
   - Your response must not exceed **225 words**.

6. **Offensive or Inappropriate Input Handling**:  
   - If the user input is disrespectful, reply with professionalism and request respectful interaction.

Context:
{context}

Question:
{user_input}
""")]

    response = llm.stream(message)
    async def stream_response():
        for chunk in response:
            yield chunk.content

    return StreamingResponse(stream_response(), media_type="text/plain")
    
@app.get("/")
async def root():
    return JSONResponse(content="Connection Successful")

@app.post("/chat")
async def chat(data: dict = Body(...)):
    document_id = data.get("document_id")
    bot_name = data.get("name")
    user_input = data.get("user_input")

    return generate_chat_response(document_id, bot_name, user_input)