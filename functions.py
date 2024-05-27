from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup
import os

OPEN_AI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(openai_api_key=OPEN_AI_API_KEY, temperature=0)
#llm = ChatOpenAI(streaming=True,callbacks=[StreamingStdOutCallbackHandler()], temperature=0, openai_api_key=OPEN_AI_API_KEY,)

vectorstore = None
qa = None

def load_new_kb(url):
    global kb_index
    global vectorstore
    global retriever
    global qa

 
    print ("ooo", url)
    docs = []
    documents = []
    for url1 in url:
        print(f"Loading documents from {url1}")
        loader = RecursiveUrlLoader(
            url=url1, 
            max_depth=3, 
            #extractor=lambda x: BeautifulSoup(x, "html.parser").text,
            extractor=lambda x: BeautifulSoup(x, "lxml").text,
            use_async=True,
            timeout=90,
            prevent_outside=True,
        )
        loader.headers = {"User-Agent": "Mozilla/5.0"}
        loader.requests_kwargs = {'verify':False}

        temp_docs = loader.load()
        temp_docs = [doc for i, doc in enumerate(temp_docs) if doc not in temp_docs[:i]]
        #print("######### Document info #################", temp_docs)
        #should check for null metedata here - checked later in the function
        docs += temp_docs

    #print("######### Document Lenght #################", len(docs))
    #print("######### FULL Document #################", docs)

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs_transformed)

    docs_transformed = splits
    # log urls and one document
    all_dataset_urls = [doc.metadata["source"] for doc in docs_transformed]
    # remove duplicates
    all_dataset_urls = list(set(all_dataset_urls))
    main_urls_str = "\n".join(all_dataset_urls)
    print(f"Loaded documents from these urls: {main_urls_str}")
    #print(f"Here is one document: {docs_transformed[3].page_content}")
    
    # Filter out documents with None metadata values - chroma db complain about it
    docs_transformed = [doc for doc in docs_transformed if all(value is not None for value in doc.metadata.values())]
    vectorstore = Chroma.from_documents(documents=docs_transformed, embedding=OpenAIEmbeddings(), persist_directory="./kdb")

    retriever = vectorstore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever)
    return True

def load_existing_kb():
    global vectorstore
    global retriever
    global qa
    # Check if the directory exists
    embedding=OpenAIEmbeddings()
    if os.path.exists("./kdb"):
        vectorstore = Chroma (persist_directory="./kdb", embedding_function=embedding)
        retriever = vectorstore.as_retriever()
        #qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever)
        qa = get_retrieval_qa()
    return True


def get_template():
    templateIdVerse = """
    You are a IAM Conference support AI with expert knowledge in Identity Management and Cyber Security. Your job is to assist users by providing accurate and concise answers to their questions for Idenvitverse 2024 conference. 
    When a user asks a question, you should provide a clear and helpful response based on the available information. Suggest similar questions from Identiverse conference that you can answer. Sprinkle a ✨ 🎉 🥳 🌟 👊 💥 💭 🔥 🚀  into your answers from time to time.
    Given this text extracts:
    -----
    {context}
    -----
 
    User question: "{question}"

    Response:
    """
    template = """
    You are a IAM Conference support AI with expert knowledge in Identity Management and Cyber Security. Your job is to assist users by providing accurate and concise answers to their questions for Gartner Security & Risk Management Summit 2024 (GartnerSEC) conference. 
    When a user asks a question, you should provide a clear and helpful response based on the available information. Suggest similar and relevant questions that you can answer. Sprinkle a ✨ 🎉 🥳 🌟 👊 💥 💭 🔥 🚀  into your answers from time to time.
    Given this text extracts:
    -----
    {context}
    -----
 
    User question: "{question}"

    Response:
    """
    return template


    return template

def get_prompt() -> PromptTemplate:
    prompt = PromptTemplate(
        template=get_template(),
        input_variables=["context", "question"]
    )
    return prompt


#---- same as retreival_qa_chain
def get_retrieval_qa():
    chain_type_kwargs = {"prompt": get_prompt()}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa

#---- same as answer_webUrls        
def retrieval_qa_inference( question, verbose=True):
    query = {"query": question}
    answer = qa(query)
    sources = "ToDo"
    #list_top_k_sources(answer, k=2)

    if verbose:
        print(sources)

    return answer["result"], sources

def retreival_qa_chain():
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        ##TODO: Use custom prompt
        #retriever = vectordb.as_retriever(search_kwargs={"k":4})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever)
        

def answer_webUrls(question:str) ->str:
    """
    Answer the question
    """
    answer = qa.run(question)

    return answer

def testfn():
     pass
