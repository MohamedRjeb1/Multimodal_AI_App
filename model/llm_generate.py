import tiktoken
from textLoader import documents
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.vectorstores import DocArrayInMemorySearch



load_dotenv()
embeddeding_model = "text-embedding-3-small"
encoding = tiktoken.get_encoding("cl100k_base")

db = DocArrayInMemorySearch.from_documents(documents, encoding)

retriever = db.as_retriever()






Gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv('GOOGLE_API_KEY'))





qa_stuff = RetrievalQA.from_chain_type(llm=Gemini, chain_type="stuff",
                                        retriever=retriever, 
                                        verbose=True)
query = "what is this tutorial about?"

response = qa_stuff.run(query)

print(response)
