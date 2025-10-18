from langchain.document_loaders import TextLoader

loader = TextLoader('./files/transcripts/transcript.txt')


documents = loader.load()

print(documents[0].page_content)