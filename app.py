import os
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from langchain_milvus import Milvus
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pymongo import MongoClient
from bson.objectid import ObjectId

from tools import followup_handler

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
    model="llama-3.3-70b-versatile",
    streaming=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Reusable vector store generator
def get_vector_store(document_id: str):
    return Milvus(
        collection_name=f"id_{document_id}",
        connection_args={"uri": os.getenv("ZILLIZ_URI_ENDPOINT"), "token": os.getenv("ZILLIZ_TOKEN")},
        index_params={"index_type": "IVF_PQ", "metric_type": "COSINE"},
        embedding_function=embeddings
    )

# Streaming response generator
def generate_chat_response(document_id: str, bot_name: str, user_input: str, chat_history: list = []):
    followup_tool=llm.bind_tools([followup_handler])
    followup_promopt=f"""You are a Legal AI assistant.  
You have access to 1 tool: `followup_handler`.  

You MUST CALL `followup_handler` if:  
1. The user's message is a follow-up to a previous conversation.  
2. The user's message is unclear, ambiguous, or lacks sufficient context to provide a confident answer.  

Examples:  
- User: "did you do the comedy factory project"
- Assistant: "Yes"
- User: "What was the Techstack"
- Tool → Restructured: "What was the Techstack of comedy factory project"  

Query:  
{user_input}
"""
    toolprompt={"role":"user","content":followup_promopt}
    chat_history.append(toolprompt)
    followupres=followup_tool.invoke(input=chat_history)
    chat_history.pop()
    if (followupres.tool_calls):
        user_input=followupres.tool_calls[0]['args']['query']
        print(f"Restructured Query: {user_input}")

    vector_store = get_vector_store(document_id)
    retrieved_docs = vector_store.similarity_search(
        query=user_input, k=3
    )
    
    context_blocks = [doc.page_content for doc in retrieved_docs]
    context = "\n\n".join(context_blocks)

    message = f"""
You are {bot_name}, a professional AI assistant representing our company.

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
"""

    chat_history.append({"role":"user","content":message})

    response = llm.stream(chat_history)
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
    chat_history = data.get("history", [])

    return generate_chat_response(document_id=document_id, bot_name=bot_name, user_input=user_input, chat_history=chat_history)

@app.post("/configuration")
async def config(request:dict = Body(...) ):
    id = request.get("id", None)
    try:
        client = MongoClient(os.getenv("MONGO_URI"))
        db=client[os.getenv("MONGO_DATABASE")]
        user_config= db[os.getenv("MONGO_COLLECTION")]
    except Exception as e:
        return JSONResponse(content={"error": e},status_code=500)
    if id:
        userData = user_config.find_one({
                "_id": ObjectId(id),
        })
        if userData:
            userData.pop("_id")
            return JSONResponse(content={"data": userData}, status_code=200)

    return  JSONResponse(content={"error":"Data not found"}, status_code=400)