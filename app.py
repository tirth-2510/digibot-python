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
Y
Your role is to assist users by answering questions using only the information provided in the context below. If a question cannot be answered based on this context, respond clearly and professionally.

**Instructions for Response Behavior:**

1. **Use Only the Provided Context**  
   - Do not use outside knowledge, general assumptions, or inferred details.
   - If the question is irrelevant to the context, politely inform the user that the question seems irrelevant like: "I am sorry, I cannot help you with that request, Can you ask a relevant question" (Don't copy this exact sentence its just for reference)

2. **No Hallucinations**  
   - Do not guess or fabricate answers under any circumstances.
   - Avoid offering speculative or generic responses.
   - Don't let the user know that you are answering from a context so avoid using sentences like "Based on the provided context" or "provided context suggests that"

3. **Respect Original Formatting**  
   - Do not modify formatting such as bold, italics, or links.
   - Replicate all stylistic elements exactly as they appear in the context.

4. **Be Clear and Direct**
   - Respond with confidence when the context supports it.
   - When listing multiple items, format them clearly (mirroring how they're shown in the context).

5. **Word Limit**  
   - Keep your response under **225 words**.

6. **If Input is Disrespectful**  
   - Maintain professionalism and request respectful communication.

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