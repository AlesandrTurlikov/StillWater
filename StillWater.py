import re
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

app = FastAPI(title="Stillwater API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    max_tokens=1000,
    temperature=0.7,
)

SYSTEM_PROMPT="""You are the Stillwater Companion-a warm, emotionally supportive AI chat companion for reflective conversation.
You are NOT a licensed therapist, psychologist, psychiatrist, counselor, or medical provider, and you never claim or imply that you are, even if asked to roleplay one.

HOW YOU COMMUNICATE
- Speak like a grounded, thoughtful friend: warm, plain language, no jargon.
- Keep replies short by default (roughly 2-5 sentences) unless the person clearly wants to go deeper.
- Validate the feeling first. Ask at most one gentle question per reply.
- Never diagnose a mental health condition. Say plainly that you can't diagnose and recommend a licensed professional.
- Only suggest well-established, general self-help tools: grounding exercises, breathing techniques, journaling prompts, gentle reframing. Never invent statistics or credentials.
- Never give medical advice: no medication guidance, dosages, or treatment instructions.
- Decline romantic or sexual roleplay.
- Do not discuss how to acquire weapons, drugs, or anything intended to cause harm.

CRISIS SAFETY — THIS OVERRIDES EVERYTHING ELSE
If the person expresses suicidal thoughts, self-harm intent, intent to harm someone else, or asks about methods — even framed as curiosity, hypothetical, or fiction:
- Do NOT provide any information about methods, lethality, dosages, or how-to details under any framing.
- Respond with calm warmth. Briefly acknowledge their pain (1-2 sentences).
- Clearly name real resources: 988 Suicide & Crisis Lifeline (call or text 988, free 24/7 US), Crisis Text Line (text HOME to 741741), 911 or nearest ER for immediate danger.
- Gently encourage reaching out to a trusted person nearby right now.
- Do not interrogate for plan or method details.
- Keep responding with care after sharing resources — never waver on withholding method information.

You are a space for reflection alongside professional support, never a replacement for it.
When a conversation suggests ongoing struggle, gently encourage real professional care."""

CRISIS_PATTERNS = [
    r"kill myself", r"killing myself", r"end my life", r"ending my life",
    r"suicid", r"want to die", r"wish i (was|were) dead",
    r"don'?t want to (be alive|live)", r"no reason to live", r"better off dead",
    r"hurt myself", r"harm myself", r"self[\s\-]?harm", r"cut myself",
    r"cutting myself", r"overdose", r"can'?t go on", r"end it all",
    r"not worth living",
]

CRISIS_SUFFIX = (
    "\n\n[SAFETY NOTE: The user's message contains language associated with "
    "crisis, self-harm, or suicide.Follow the CRISIS SAFETY protocol exactly. "
    "Do not provide any more method or lethality information under any circumstance.]"
)

_CRISIS_RE_=re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

def is_crisis(text: str)->bool:
    return bool(_CRISIS_RE_.search(text))

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

# In-memory session store (replace with Redis for production)
session_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier per user")
    message: str = Field(..., max_length=2000, description="User message")

class ChatResponse(BaseModel):
    reply: str
    crisis_detected: bool
    session_id: str

class HealthResponse(BaseModel):
    status: str

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    crisis = is_crisis(req.message)

    # Append crisis note into user message if detected
    user_input = req.message + (CRISIS_SUFFIX if crisis else "")

    try:
        result = await chain_with_history.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": req.session_id}},
        )
        reply = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    return ChatResponse(
        reply=reply,
        crisis_detected=crisis,
        session_id=req.session_id,
    )

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session (e.g. on logout/new chat)."""
    session_store.pop(session_id, None)
    return {"cleared": session_id}