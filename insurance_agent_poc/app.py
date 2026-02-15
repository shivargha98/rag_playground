import streamlit as st
import os
import time
import mongomock
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- CONFIGURATION ---
ST_PAGE_TITLE = "Nova Health (Gold Member) AI"
POLICY_FILENAME = "policies/gold_policy.txt"


# --- 1. SETUP: CREATE DUMMY DATA ---
def setup_mock_data():
    """Creates the policy file and seeds the mock MongoDB."""
    
    # A. Create Policy File (The Dense Gold Plan)
    if not os.path.exists(POLICY_FILENAME):
        policy_content = """
        POLICY CONTRACT: Nova Health 'GOLD' PRIVILEGE
        UIN: SHA-HLT-GLD-2025-V1 | CATEGORY: COMPREHENSIVE

        SECTION 1: OPERATIVE CLAUSE
        The Company undertakes to indemnify the Insured Person against Medically Necessary expenses.

        SECTION 2: CORE BENEFITS
        2.1. ROOM RENT ELIGIBILITY
            (a) The Policy covers expenses for a "Single Private A/C Room".
            (b) Upgrade Clause: If the Insured opts for a Suite, the Company will reimburse room charges up to the Single Private A/C Room tariff. NO Proportionate Deduction will apply to doctor's fees.

        2.2. RESTORATION OF SUM INSURED
            (a) If Base SI is exhausted, 100% restoration triggers once per year.
            (b) Restriction: Cannot be used for the SAME illness.

        SECTION 3: TREATMENT-SPECIFIC COVERAGE
        3.1. MODERN TREATMENT METHODS (Capped at 50% SI):
            i. Uterine Artery Embolization, ii. Balloon Sinuplasty, iii. Robotic Surgery.

        SECTION 4: EXCLUSIONS & WAITING PERIODS
        4.1. Standard Waiting Period: 30 days.
        4.2. Named Ailments (2 Years): Hysterectomy, Knee Replacement, Cataract.
        4.3. Pre-Existing Diseases (PED): Covered after 36 continuous months.

        SECTION 5: NETWORK & CASHLESS
        Cashless service is authorized for all hospitals listed in the Gold Network.
        
        ANNEXURE A: GOLD NETWORK HOSPITALS
        1. [HOSP-101] St. Mary’s Cardiac Institute - Bangalore
        2. [HOSP-102] Apollo Indraprastha - New Delhi
        3. [HOSP-110] Max Super Speciality - Mumbai 
        4. [HOSP-120] Ruby Hall Clinic - Pune
        """
        with open(POLICY_FILENAME, "w") as f:
            f.write(policy_content)

    # B. Seed Mock MongoDB
    client = mongomock.MongoClient()
    db = client["health_insurance_db"]
    
    # Seed Member (Our User)
    if db.members.count_documents({}) == 0:
        db.members.insert_one({
            "member_id": "MEM_GOLD_01",
            "name": "Priya Mehta",
            "plan_type": "Gold Privilege",
            "base_sum_insured": 1000000,
            "policy_start_date": "2023-01-01",
            "pre_existing_diseases": ["Hypertension"],
            "cumulative_bonus": 50000,
            "claims_made_this_year": False
        })

    # Seed Providers
    if db.providers.count_documents({}) == 0:
        db.providers.insert_many([
            {"id": "HOSP-102", "name": "Apollo Indraprastha", "city": "New Delhi", "tier": "Tier 1", "cashless": True},
            {"id": "HOSP-120", "name": "Ruby Hall Clinic", "city": "Pune", "tier": "Tier 2", "cashless": True},
            {"id": "HOSP-999", "name": "City Local Clinic", "city": "Pune", "tier": "Tier 3", "cashless": False}
        ])
    
    return db

# --- 2. AGENT LOGIC ---

class AgenticSystem:
    def __init__(self, api_key):
        self.db = setup_mock_data()
        self.user_id = "MEM_GOLD_01"  # Hardcoded authenticated user
        
        # Initialize Gemini Components
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key="AIzaSyCeyJm3HWTh0EPQW2yPjaFMyljrTlOlZL0", 
            temperature=0
        )
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize RAG
        self._init_rag()

    def _init_rag(self):
        loader = TextLoader(POLICY_FILENAME)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever()

    def get_member_info(self):
        """Database Agent: Fetches Member Details"""
        member = self.db.members.find_one({"member_id": self.user_id})
        del member['_id'] # Remove ObjectID for cleaner printing
        return str(member)

    def check_provider(self, query):
        """Database Agent: Searches for Hospitals"""
        # Simple keyword search in mock DB
        providers = list(self.db.providers.find())
        matches = [p for p in providers if p['name'].lower() in query.lower()]
        if matches:
            return f"Found Provider(s): {matches}"
        return "No specific network hospital found in database match."

    def get_policy_info(self, query):
        """Policy Agent: RAG Search"""
        docs = self.retriever.get_relevant_documents(query)
        return "\n".join([d.page_content for d in docs])

    def orchestrate(self, user_query):
        """The Main Brain: Decides what context is needed."""
        
        gathered_context = []
        logs = []

        # Step 1: Always get Member Context (Since we know who is logged in)
        with st.status("Thinking...", expanded=True) as status:
            st.write("🔍 Authenticating Member...")
            member_data = self.get_member_info()
            gathered_context.append(f"MEMBER DATA: {member_data}")
            logs.append("Fetched Member Data")
            time.sleep(0.5) # UX Pause

            # Step 2: Check if we need Policy Info
            st.write("📖 Consulting Policy Documents...")
            policy_data = self.get_policy_info(user_query)
            gathered_context.append(f"POLICY CLAUSES: {policy_data}")
            logs.append("Fetched Policy Clauses")
            time.sleep(0.5)

            # Step 3: Check if we need Provider Info
            if "hospital" in user_query.lower() or "clinic" in user_query.lower() or "where" in user_query.lower():
                st.write("🏥 Checking Network Database...")
                provider_data = self.check_provider(user_query)
                gathered_context.append(f"PROVIDER DB: {provider_data}")
                logs.append("Fetched Provider Data")
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

        # Step 4: Final Synthesis
        final_prompt = f"""
        You are a helpful Support Agent for Nova Health.
        The user is a GOLD Plan member.
        
        CONTEXT:
        {chr(10).join(gathered_context)}
        
        USER QUESTION: {user_query}
        
        INSTRUCTIONS:
        1. Answer strictly based on the Policy Clauses and Member Data provided.
        2. If the user asks about a specific hospital, check the Provider DB context.
        3. Be polite and professional.
        4. If the policy mentions a waiting period, check the member's 'policy_start_date' (Current Date: Jan 2026) to see if they have crossed it.
        """
        
        response = self.llm.invoke(final_prompt)
        return response.content

# --- 3. STREAMLIT UI ---

st.set_page_config(page_title="Insurance Agent", layout="centered")

st.title("🛡️ Member Advantage AI")
#st.subheader("Gold Member Concierge")

# API Key Handling
if "api_key" not in st.session_state:
    st.session_state.api_key = "AIzaSyCeyJm3HWTh0EPQW2yPjaFMyljrTlOlZL0"

with st.sidebar:
    # st.header("Settings")
    # api_input = st.text_input("Enter Gemini API Key", type="password")
    # if api_input:
    #     st.session_state.api_key = api_input
    
    st.divider()
    st.info("**Welcome Priya Mehta **")
    st.info("Try asking:\n- 'Is Apollo covered?'\n- 'Does my plan cover robotic surgery?'\n- 'Can I claim for Cataract now?'")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# if not st.session_state.api_key:
#     st.warning("Please enter your Google Gemini API Key in the sidebar to continue.")
#     st.stop()

# Initialize System
if "agent_system" not in st.session_state:
    st.session_state.agent_system = AgenticSystem(st.session_state.api_key)

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("How can I help you with your policy?"):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Response
    with st.chat_message("assistant"):
        response = st.session_state.agent_system.orchestrate(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})