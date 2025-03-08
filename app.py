# ------------------------------
# 1. Package Installation (if needed)
# ------------------------------
#!pip install langchain langchain_community pymupdf pypdf openai langchain_openai

# ------------------------------
# 2. Imports
# ------------------------------
import os
import concurrent.futures
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
# Load environment variables from a .env file.
from dotenv import load_dotenv

# ------------------------------
# 3. Environment and Model Initialization
# ------------------------------
load_dotenv()

# Set your OpenAI API key as an environment variable.
#os.environ["OPENAI_API_KEY"] = "sk-<your-openai-api-key>"

# Initialize the general language model for question-answering.
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Initialize a second language model specifically for assessing document relevancy.
llm_relevancy = ChatOpenAI(
    model="o3-mini",
    reasoning_effort="medium",
    max_tokens=3000,
)

# ------------------------------
# 4. Prompt Templates
# ------------------------------

# System prompt to guide the relevancy extraction process.
REAG_SYSTEM_PROMPT = """
# Role and Objective
You are an intelligent knowledge retrieval assistant. Your task is to analyze provided documents or URLs to extract the most relevant information for user queries.

# Instructions
1. Analyze the user's query carefully to identify key concepts and requirements.
2. Search through the provided sources for relevant information and output the relevant parts in the 'content' field.
3. If you cannot find the necessary information in the documents, return 'isIrrelevant: true', otherwise return 'isIrrelevant: false'.

# Constraints
- Do not make assumptions beyond available data
- Clearly indicate if relevant information is not found
- Maintain objectivity in source selection
"""

# Prompt template for the retrieval-augmented generation (RAG) chain.
rag_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

# ------------------------------
# 5. Schema Definitions and JSON Parser Setup
# ------------------------------

# Define a schema for the expected JSON response from the relevancy analysis.
class ResponseSchema(BaseModel):
    content: str = Field(..., description="The page content of the document that is relevant or sufficient to answer the question asked")
    reasoning: str = Field(..., description="The reasoning for selecting the page content with respect to the question asked")
    is_irrelevant: bool = Field(..., description="True if the document content is not sufficient or relevant to answer the question, otherwise False")

# Wrapper model for the relevancy response.
class RelevancySchemaMessage(BaseModel):
    source: ResponseSchema

# Create a JSON output parser using the defined schema.
relevancy_parser = JsonOutputParser(pydantic_object=RelevancySchemaMessage)

# ------------------------------
# 6. Helper Functions
# ------------------------------

# Format a Document into a human-readable string that includes metadata.
def format_doc(doc: Document) -> str:
    return f"Document_Title: {doc.metadata['title']}\nPage: {doc.metadata['page']}\nContent: {doc.page_content}"

# Define a helper function to process a single document.
def process_doc(doc: Document, question: str):
    # Format the document details.
    formatted_document = format_doc(doc)
    # Combine the system prompt with the document details.
    system = f"{REAG_SYSTEM_PROMPT}\n\n# Available source\n\n{formatted_document}"
    # Create a prompt instructing the model to determine the relevancy.
    prompt = f"""Determine if the 'Avaiable source' content supplied is sufficient and relevant to ANSWER the QUESTION asked.
    QUESTION: {question}
    #INSTRUCTIONS TO FOLLOW
    1. Analyze the context provided thoroughly to check its relevancy to help formulize a response for the QUESTION asked.
    2. STRICTLY PROVIDE THE RESPONSE IN A JSON STRUCTURE AS DESCRIBED BELOW:
        ```json
           {{"content":<<The page content of the document that is relevant or sufficient to answer the question asked>>,
             "reasoning":<<The reasoning for selecting the page content with respect to the question asked>>,
             "is_irrelevant":<<Specify 'True' if the content in the document is not sufficient or relevant. Specify 'False' if the page content is sufficient to answer the QUESTION>>
             }}
        ```
     """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    # Invoke the relevancy language model.
    response = llm_relevancy.invoke(messages)
    #print(response.content)  # Debug output to review model's response.
    # Parse the JSON response.
    formatted_response = relevancy_parser.parse(response.content)
    return formatted_response

# Extract relevant context from the provided documents given a question, using parallel execution.
def extract_relevant_context(question, documents):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all document processing tasks concurrently.
        futures = [executor.submit(process_doc, doc, question) for doc in documents]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing document: {e}")
    # Collect content from documents that are relevant.
    final_context = [
        item['content']
        for item in results
        if str(item['is_irrelevant']).lower() == 'false'
    ]
    return final_context

# Generate the final answer using the RAG approach.
def generate_response(question, final_context):
    # Create the prompt using the provided question and the retrieved context.
    prompt = PromptTemplate(template=rag_prompt, input_variables=["question", "context"])
    # Chain the prompt with the general language model.
    chain = prompt | llm
    # Invoke the chain to get the answer.
    response = chain.invoke({"question": question, "context": final_context})
    answer = response.content.split("\n\n")[-1]
    return answer

# ------------------------------
# 7. Main Execution Block
# ------------------------------
if __name__ == "__main__":
    # Load the document from the given PDF URL.
    file_path = "https://www.binasss.sa.cr/int23/8.pdf"
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")
    #print("Metadata of the first document:", docs[0].metadata)

    # Example 1: Answer the question "What is Fibromyalgia?"
    question1 = "What is Fibromyalgia?"
    context1 = extract_relevant_context(question1, docs)
    print(f"Extracted {len(context1)} relevant context segments for the first question.")
    answer1 = generate_response(question1, context1)

    # Print the results.
    print("\n\nQuestion 1:", question1)
    print("Answer to the first question:", answer1)

    # Example 2: Answer the question "What are the causes of Fibromyalgia?"
    question2 = "What are the causes of Fibromyalgia?"
    context2 = extract_relevant_context(question2, docs)
    answer2 = generate_response(question2, context2)
    
    # Print the results.
    print("\nQuestion 2:", question2)
    print("Answer to the second question:", answer2)