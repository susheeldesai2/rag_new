#1. Extract/Load the PDF in the System
#2. Chunk the PDF
#3. Embed and Store in Vector DB
#4. User Asks Question
#5. Similaroty Search
#6. Retreive Topk Results, Combine with question and Ask llm question
#7. LLM output
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
load_dotenv()


#Step1: Read Text from PDF File

text=""
pdf_reader=PdfReader("incorrect_facts.pdf")

for page in pdf_reader.pages:
    text+=page.extract_text() + "\n"

#Step2: Chunk the Extracted Text

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_text(text)

#printing the chunks to view
# for i in range(len(chunks)):
#     print(f"Chunk {i+1}: {chunks[i]}")

#Initialize Vector Database
pc = Pinecone()

index_name = "rag"
index = pc.Index(index_name)
#creating index and then commenting once the index is created
# pc.create_index(
#     name=index_name,
#     dimension=384, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

#Select embedding model

embedding_model = SentenceTransformer("BAAI/bge-small-en")

#Embed and Store in Vector DB and comment once the database is crated in pinecone
#print(type(chunks))
# for i, chunk in enumerate(chunks):
#     chunk_embedding=embedding_model.encode(chunk, normalize_embeddings=True)
#     index.upsert([(str(i+1),chunk_embedding.tolist(),{"chunk":chunk})])

#User query
llm=ChatGroq(temperature=0,model="llama3-70b-8192")

query = "How do Birds Migrate"
question_embedding = embedding_model.encode(query, normalize_embeddings=True)  # Fix applied

#Retrive top K results
result=index.query(vector=question_embedding.tolist(),top_k=3,include_metadata=True)


augmented_text="\n\n".join([match.metadata["chunk"] for match in result.matches])
#print(augmented_text)


#Creating chatbot

prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the context provided to answer the question accurately. Only use this context, not your knowledge.\n\n"
                "Context:{context}"
                "Question:{question}"
                
    )

chain = prompt | llm 

response = chain.invoke({"context":augmented_text, "question": query})


print(response.content)
