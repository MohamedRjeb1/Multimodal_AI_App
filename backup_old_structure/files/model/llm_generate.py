import os
from dotenv import load_dotenv

from textLoader import documents

# LangChain + Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch

# ta variable `documents` : liste de Document (LangChain) ou objets similaires
# from textLoader import documents

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your .env")



embeddings = GoogleGenerativeAIEmbeddings()  




db = DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)

# Créer un retriever à partir du vectorstore
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})




# 4) Initialiser Gemini (Chat) — option transport="rest" si besoin
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)

# 5) Construire la pipeline RAG (RetrievalQA)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",   # ou "map_reduce", "refine" selon la taille / préférences
    retriever=retriever,
    verbose=True,
    
)

# 6) Lancer une requête
query = "what is this tutorial about?"
response = qa.run(query)
print(response)
