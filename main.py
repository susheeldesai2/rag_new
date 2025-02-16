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
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
load_dotenv()


#Step1: Read Text from PDF File

text=""
pdf_reader=PdfReader("incorrect_facts.pdf")

for page in pdf_reader.pages:
    text+=page.extract_text() + "\n"


#Step2: Chunk the Extracted Text

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50, length_function=len)
chunks = text_splitter.split_text(text)


#Initialize Vector Database

pc = Pinecone()

index_name = "rag"
index = pc.Index(index_name)

# pc.create_index(
#     name=index_name,
#     dimension=1536, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

#Select Embedding Model

embedding=OpenAIEmbeddings()

#Step 3: Convert chunks to Embedding and Store in Vector Database
# for i, chunk in enumerate(chunks):
#     chunk_embedding=embedding.embed_query(chunk)
#     index.upsert([(str(i+1),chunk_embedding,{"chunk":chunk,})])

#User Query

llm=ChatOpenAI(model="gpt-4",temperature=0)


query="How do Birds Migrate"
question_embedding=embedding.embed_query(query)

#Retrieve Topk Results

result=index.query(vector=question_embedding,top_k=3,include_metadata=True)


augmented_text="\n\n".join([match.metadata["chunk"] for match in result.matches])



#Creating the CHatbot

prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are a helpful assistant. Use the context provided to answer the question accurately. Only use this context, not your knowledge.\n\n"
                "Context:{context}"
                "Question:{question}"
                
    )

chain = prompt | llm 

response = chain.invoke({"context":augmented_text, "question": query})


print(response.content)
