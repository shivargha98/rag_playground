import streamlit as st
import os
import time
import io

import mongomock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from streamlit_mic_recorder import mic_recorder
from elevenlabs.client import ElevenLabs

# --- CONFIGURATION ---
ST_PAGE_TITLE = "Nova Health Voice AI"
POLICY_FILENAME = "policies/gold_policy.txt"
# Standard ElevenLabs Voice ID (Rachel)
VOICE_ID = "21m00Tcm4TlvDq8ikWAM" 

st.set_page_config(
    page_title=ST_PAGE_TITLE,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 1. MOCK DATA SETUP ---
def setup_mock_data():
    if not os.path.exists(POLICY_FILENAME):
        os.makedirs("policies", exist_ok=True)
        # (Same dense policy content as before)
        policy_content = """
        POLICY CONTRACT: Nova Health 'GOLD' PRIVILEGE
        SECTION 1: OPERATIVE CLAUSE: Indemnity for Medically Necessary expenses.
        SECTION 2: ROOM RENT: Covers "Single Private A/C Room".
        SECTION 3: MODERN TREATMENT: Robotic Surgery, Balloon Sinuplasty capped at 50% SI.
        SECTION 4: WAITING PERIODS: 30 days standard. 2 Years for Cataract.
        SECTION 5: NETWORK: Cashless at Apollo Indraprastha, Ruby Hall Clinic.
        """
        with open(POLICY_FILENAME, "w") as f:
            f.write(policy_content)

    client = mongomock.MongoClient()
    db = client["health_insurance_db"]
    
    # Ensure ID match
    if db.members.count_documents({}) == 0:
        db.members.insert_one({
            "member_id": "MEM_GOLD_01",
            "name": "Priya Mehta",
            "plan_type": "Gold Privilege",
            "policy_start_date": "2023-01-01"
        })
    
    if db.providers.count_documents({}) == 0:
        db.providers.insert_many([
            {"name": "Apollo Indraprastha", "city": "New Delhi", "cashless": True},
            {"name": "Ruby Hall Clinic", "city": "Pune", "cashless": True}
        ])
    return db

# --- 2. INTELLIGENCE LAYER (Gemini + ElevenLabs) ---
class AgenticSystem:
    def __init__(self, google_key, eleven_key):
        self.db = setup_mock_data()
        self.user_id = "MEM_GOLD_01"
        
        # 1. Google Gemini (The Brain)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key="AIzaSyAMm1jvDyuSf7026ZxLMyZDmAxM71_rnQw",
            temperature=0
        )
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # 2. ElevenLabs (The Mouth & Ears)
        self.eleven = ElevenLabs(api_key="sk_37d80bdf4034fc03f39c1109828397d20e83fcc27d42e2e0")
        
        self._init_rag()

    def _init_rag(self):
        loader = TextLoader(POLICY_FILENAME)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vector_store = FAISS.from_documents(splitter.split_documents(docs), self.embeddings)
        self.retriever = self.vector_store.as_retriever()

    def transcribe_audio(self, audio_bytes):
        """ElevenLabs Scribe v2 STT"""
        # Convert bytes to file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav" # Scribe needs a filename hint
        
        try:
            transcription = self.eleven.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2" 
            )
            print('audio_generated')
            print(transcription.text)
            return transcription.text
        except Exception as e:
            return f"Error transcribing: {str(e)}"

    def generate_audio(self, text):
        """ElevenLabs TTS"""
        try:
            audio_gen = self.eleven.text_to_speech.convert(
                text=text,
                voice_id="1qEiC6qsybMkmnNdVMbK",
                model_id="eleven_multilingual_v2"
            )
            # Consume generator to bytes
            audio_bytes = b"".join(chunk for chunk in audio_gen)
            return audio_bytes
        except Exception as e:
            st.error(f"TTS Error: {e}")
            return None

    def orchestrate(self, user_query):
        # (Simplified Orchestrator for brevity)
        docs = self.retriever.get_relevant_documents(user_query)
        context = "\n".join([d.page_content for d in docs])
        
        providers = list(self.db.providers.find())
        prov_str = str([p['name'] for p in providers])
        
        prompt = f"""
        You are a helpful Support Agent for Nova Health.
        The user is a GOLD Plan member.

        INSTRUCTIONS:
        1. Answer strictly based on the Policy Clauses and Member Data provided.
        2. If the user asks about a specific hospital, check the Provider DB context.
        3. Be polite and professional.
        4. If the policy mentions a waiting period, check the member's 'policy_start_date' (Current Date: Jan 2026) to see if they have crossed it.
        Context: {context}
        Providers: {prov_str}
        User: {user_query}
        Answer:
        """
        response_text = self.llm.invoke(prompt).content
        print('agent_resp:',response_text)
        return response_text

# --- 3. UI LAYOUT ---

with st.sidebar:
    st.title("🎙️ Nova Voice")
    st.markdown("### Settings")
    
    # Secure Key Entry
    g_key = "AIzaSyAMm1jvDyuSf7026ZxLMyZDmAxM71_rnQw"
    e_key = "sk_37d80bdf4034fc03f39c1109828397d20e83fcc27d42e2e0"
    
    st.divider()
    
    st.markdown("### 🗣️ Push to Speak")
    # Microphone Component
    audio_data = mic_recorder(
        start_prompt="Record",
        stop_prompt="Stop",
        key="recorder",
        #just_once=True,
        use_container_width=True
    )

# Initialize Session
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi Priya! I'm listening. How can I help?"}]

# Initialize System
if g_key and e_key:
    if "agent" not in st.session_state:
        st.session_state.agent = AgenticSystem(g_key, e_key)
else:
    st.warning("Please enter both API Keys in the sidebar to activate the agent.")

# --- MAIN CHAT LOGIC ---
chat_container = st.container()

# A. HANDLE VOICE INPUT
if audio_data is not None and "agent" in st.session_state:
    # 1. Transcribe (Scribe v2)
    with st.spinner("Transcribing with Scribe..."):
        #time.sleep(1)
        text_input = st.session_state.agent.transcribe_audio(audio_data['bytes'])
    
        # 2. Add User Message
        st.session_state.messages.append({"role": "user", "content": text_input})

        # 3. Get Response
        response_text = st.session_state.agent.orchestrate(text_input)
        #print(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("user"):
        st.markdown(text_input)
        
    with st.chat_message("assistant"):
        st.markdown(response_text)
        
    #4. Generate Audio (ElevenLabs TTS)
    audio_bytes = st.session_state.agent.generate_audio(response_text)
    st.session_state.latest_audio = audio_bytes
    st.audio(st.session_state.latest_audio, format="audio/mp3", autoplay=True)
    
    # Force Rerun to display messages
    #st.rerun()

# B. RENDER MESSAGES
# with chat_container:
#     for i, msg in enumerate(st.session_state.messages):
#         if msg["role"] == "user":
#             st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="bot-bubble">🛡️ <b>Nova:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            
#             # If this is the very last message and we have audio for it, play it
#             if i == len(st.session_state.messages) - 1 and "latest_audio" in st.session_state:
#                 st.audio(st.session_state.latest_audio, format="audio/mp3", autoplay=True)

# C. TEXT FALLBACK
# Even in a voice app, keeping text input is good UX
text_prompt = st.chat_input("Or type your question here...")
if text_prompt and "agent" in st.session_state:
    st.session_state.messages.append({"role": "user", "content": text_prompt})
    response = st.session_state.agent.orchestrate(text_prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Also generate audio for text inputs
    audio_bytes = st.session_state.agent.generate_audio(response)
    st.session_state.latest_audio = audio_bytes
    st.rerun()