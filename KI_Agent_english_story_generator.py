#!/usr/bin/env python
# coding: utf-8

# # English Learners' Short Story Generator


# --------------------
# ### Öffne User Interface im Browser mit Streamlit
# --------------------
import streamlit as st

st.markdown(
    "<h1 style='text-align: center; color: darkblue; font-family: Arial;'>English Learners' Short Story Generator</h1>",
    unsafe_allow_html=True
)

# Eingaben des Users
topic = st.text_input("📝 What should your story be about? (e.g. friendship, adventure, or mystery)")
book = st.text_input("📚 From which book (title and author) should the story take ideas or vocabulary?")

# Prompt-Values zusammenbauen
prompt_values = {
    "topic": topic,
    "book": book
}

# --------------------
# ### LLM-Setup
# --------------------
import os
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
load_dotenv()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") #Umgebungsvariable aus .env-Datei laden

llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini") #LangChain-Wrapper für OpenAI

# --------------------
# ### LangSmith zur Agenten-Evaluierung
# --------------------
from langsmith import Client

# aus .env holen
os.environ["LANGSMITH_API_KEY"] = os.getenv("VAWI_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com" # EU-Endpoint wählen
os.environ["LANGCHAIN_PROJECT"] = "English Learners' Short Story Generator"
client = Client()

# --------------------
# ### Vokabeln laden (als Wissenskorpus für das Vokabel-RAG-Tool)
# --------------------
#Step 1: Text laden via langchain csvloader mit Metadaten
from langchain_community.document_loaders.csv_loader import CSVLoader 

# Absoluter Pfad zur CSV, relativ zu .py-Datei
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "vokabeln_mit_themen.csv")

loader = CSVLoader(file_path=csv_path, csv_args={"delimiter": ";"}, metadata_columns=["topic"]) #, "unit", "learning_year"
docs = loader.load()

# Metadaten um "Noise" bereinigen: nur 'topic' behalten
for doc in docs:
    topic_value = doc.metadata.get("topic")
    doc.metadata = {"topic": topic_value}

# Test: erstes Dokument ausgeben
#print("CONTENT:", docs[0].page_content)
#print("METADATA:", docs[0].metadata)


#Step 2: Semantisches Embedding und Vektorisierung des Textes
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

@st.cache_resource #Embeddings und Vektordaten in Streamlit cachen, damit diese Schritte nur einmal ausgeführt werden 
def load_vectorstore(_docs):
    # Embeddings mit SentenceTransformer laden (nur einmal)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Text splitten, um Größe des Kontextfensters nicht zu überfrachten 
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=50, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs)

    # FAISS-Vektorspeicher erstellen (ebenfalls gecacht)
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding_model)
    return vectorstore

# FAISS Vektorstore-Objekt laden
vectorstore = load_vectorstore(docs)

#Step 3: Retriever-Objekt instantiieren
#sucht die Top 50 Ergebnisse nach semantischer Ähnlichkeit
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50})

# --------------------
# ### Vokabel-RAG-Tool
# --------------------
from langchain_core.tools import tool
from collections import defaultdict

@tool #LangChain Tool-decorator
def retrieve_vocab(topic: str = ""):
    """Retrieve vocabulary and group by topic.""" #Funktionsbeschreibung wichtig für das LLM

    results = retriever.invoke(topic)  #Durchführen der Suche nach passenden/ähnlichen Vokabeln

    # Gruppiere Vokabeln pro Thema
    grouped_vocab = defaultdict(list)
    for doc in results:
        # Metadata als String
        metadata_str = ", ".join(str(v) for v in doc.metadata.values())

        # Vokabeltext extrahieren
        text = doc.page_content
        if text.lower().startswith("vocabulary:"):
            text = text[len("vocabulary:"):].strip()

        grouped_vocab[metadata_str].append(text)

    # Ausgabe: ein String pro Thema mit allen Vokabeln
    output = [
        f"[{topic}] " + ", ".join(vocabs)
        for topic, vocabs in grouped_vocab.items()
    ]

    return output

# --------------------
# ### Wikipedia-Tool
# --------------------
# Externes Wikipedia-Modul, um nach Inhalten auf Wikipedia zu suchen
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Wikipedia API wrapper initialisieren, um Top 1 Ergebnis auszugeben
api_wrapper = WikipediaAPIWrapper(top_k_results=1)

# Erzeuge Wikipedia-Such-Tool mit Hilfe des zuvor erstellten Wrappers
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# --------------------
# ### Agenten-Orchestrierung mit Hilfe von LangGraph's High-level Syntax
# --------------------
#LangGraph's High-level-Syntax abstrahiert das Erstellen von dedizierten Edges und Nodes
#Hier: Erstellen eines übergeordneten Supervisor-Agenten und zwei untergeordneter Agenten, die weiteren Kontext (relevante Vokabeln und Wikipedia-Eintrag) bereitstellen.

#Erstelle zwei Agenten: einen Vokabel-Retrieval Agenten ("vocabulary_retrieval_agent") und einen Wikipedia Agenten ("wikipedia_agent"). 
from langgraph.prebuilt import create_react_agent

# --------------------
# ### Vokabel Agent
# --------------------
#Prompt-Template für den Vokabel-Retrieval-Agenten
vocabulary_retrieval_agent_prompt_template = (
"You are a vocabulary-retrieval agent.\n\n"
"INSTRUCTIONS:\n"
"- Retrieve relevant vocabulary for generating a short story.\n"
"- After you're done with your tasks, respond to the supervisor directly\n"
"- Respond ONLY with the results of your work, do NOT include ANY other text.")


#Vokabel-Agent erstellen
vocabulary_retrieval_agent = create_react_agent(
    llm,
    tools=[retrieve_vocab],
    prompt=vocabulary_retrieval_agent_prompt_template,
    name="vocabulary_retriever"
)

# --------------------
# ### Wikipedia Agent
# --------------------
#Prompt für den Wikipedia Agenten
from langchain.prompts import PromptTemplate

wikipedia_prompt_template = PromptTemplate(
    template=(
        "You are a wikipedia search agent.\n\n"
        "INSTRUCTIONS:\n"
        "- ONLY use '{book}' as the search term.\n"
        "- Ignore all other user input.\n"
        "- Return only the genre, content and main characters."
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    input_variables=["book"]
)

# Vor Übergabe an den Agenten Prompt mit dem aktuellen Inhalt der Variabe füllen
wikipedia_prompt = wikipedia_prompt_template.format(book=prompt_values["book"])

#Wikipedia-Agent erstellen
wikipedia_agent = create_react_agent(
    llm,
    [wikipedia_tool],
    prompt=wikipedia_prompt,
    name="wikipedia_search"
)

# --------------------
# ### Supervisor Agent
# --------------------
#Prompt für den Supervisor-Agenten
supervisor_prompt = """You are a supervisor agent. 
You control two worker agents:
- wikipedia_search: Only uses the variable {book} as input
- vocabulary_retriever: Only uses the variables {topic} and {book} as input

Route tasks to the correct agent by giving them ONLY the variable(s) they need.
Do not forward the full user input to sub-agents.

Your goal is to support English learners by writing **short, entertaining and motivating stories** in English.

Here are further rules you must follow:

1. **Book & Plot**: 
   - The story should be based on the plot of the book specified by the user.
   - The main characters of the book must appear in the short story.
   - Use the topic given by the user to write a matching story.

2. **Vocabulary & Level**:
   - Use only A1-A2 level English grammar. If in doubt, use language and grammar which is easy to understand.
   - Use **ONLY THE VOCABULARY THAT YOU RETRIEVE FROM THE vocabulary_retrieval_agent**.
   - Do NOT use any other vocabulary.

3. **Language**:
   - Write the story in English.

4. **Output**:
   - 1. Present the short story as continuous text with about 300 words. 
   - 2. Below the story, list all vocabulary from the RAG which was used in the story with their German translation.
   - 3. Always show the short story and below show the vocabulary.
"""

# Prompt mit Werten befüllen
filled_supervisor_prompt = supervisor_prompt.format(**prompt_values)


#Supervisor-Agent erstellen, der Aufgaben an 'retriever_agent` und `wikipedia_agent` verteilt
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver 

config = {"configurable": {"thread_id": "1", "user_id": "1"}} #notwendig, um Ausführungsschritte zu handeln
checkpointer = InMemorySaver() #Kurzzeitspeicher, um Inputs für die Dauer der Ausführung zu behalten

supervisor = create_supervisor(
    model=llm,
    agents=[vocabulary_retrieval_agent, wikipedia_agent],
    prompt= filled_supervisor_prompt, 
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile(checkpointer=checkpointer)

# --------------------
# ### Safety-Check (Guardrails) für kindgerechte Eingaben
# --------------------
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

safety_prompt = ChatPromptTemplate.from_template("""
You are a safety checker for children's content.
Your job is to determine whether the following topic and book input are appropriate for children under 14.

Reject or flag anything that includes:
- Violence, hate, discrimination
- Drugs, alcohol, or sexual content
- Political or religious extremism

Topic: {topic}
Book: {book}

Return ONLY one of the following:
- "SAFE" if both inputs are appropriate for children
- "UNSAFE" if not
""")

safety_chain = LLMChain(llm=llm, prompt=safety_prompt)

# --------------------
# ### Supervisor-Agent mit User Prompt aufrufen (Verwendung der oben definierten Guardrails als Policy Wrapper um das LLM)
# --------------------
if st.button("✨ Generate Story"):
    if topic and book:
        with st.spinner("Checking content safety..."):
            safety_result = safety_chain.run({"topic": topic, "book": book}).strip()

        if safety_result != "SAFE":
            st.error("⚠️ The topic or book is not suitable for child-friendly stories. Please choose another one.")
        else:
            with st.spinner("Generating story..."):
                result = supervisor.invoke(
                    {
                        "messages": [("user", f"Please write a story about '{topic}' in the style of {book} with vocabulary from the vocabulary_retrieval_agent.")],
                        "book": book,
                        "topic": topic
                    },
                    config=config
                )

            last_msg = result["messages"][-1]
            story_content = getattr(last_msg, "content", "No content available")

            st.subheader("Generated Story")
            st.write(story_content)

    else:
        st.warning("Bitte sowohl *Topic* als auch *Book* eingeben.")
