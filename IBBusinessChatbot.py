import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

import nest_asyncio

from llama_parse import LlamaParse

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key https://makersuite.google.com/app/apikey.

2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")



# This is the first API key input; no need to repeat it in the main function.
name = st.text_input("Enter your name", type="password", key="user_question")

# Initialize session state variables
if 'vectorstore_created' not in st.session_state:
    st.session_state.vectorstore_created = False

if 'history_aware_retriever_created' not in st.session_state:
    st.session_state.history_aware_retriever = None


if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None

def get_pdf_text():

    nest_asyncio.apply()

    parser = LlamaParse(
        api_key="llx-OmxZDgHY0WOPAYcxtKPD33KPQ56KPys4MGeYyVIlca2tqs4v",  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )

# sync
documents = parser.load_data("ocr_unit1.pdf")

text = '\n\n'.join([d.text for d in documents])

session_init = False

def create_vector_store(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    retriever.save_local("chroma_retriever")

    st.session_state.vectorstore_created = True


def create_retriever_with_history():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    retriever = Chroma.load_local("chroma_retriever", OpenAIEmbeddings())
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    st.session_state.history_aware_retriever = history_aware_retriever

    # return history_aware_retriever


def get_conversational_chain(history_aware_retriever):
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key = "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", temperature=0)

    system_prompt = (
    "You are an assistant for question-answering tasks related to IB Business. "
    "You function as an automated tutor and your goal is to help students better understand content. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. "
    "\n\n"
    "{context}"
)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    st.session_state.conversational_rag_chain = conversational_rag_chain










    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # return chain


def validate_document_viability():
    # We use this to validate if the given documents are appropriate to answer the question or not
    pass

def user_input(user_question, conversational_rag_chain):
    response = st.session_state.conversational_rag_chain.invoke(
                    {"input": user_question},
                    config={
                        "configurable": {"session_id": "abc123"}
                    },  # constructs a key "abc123" in `store`.
                )["answer"]

    


    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    # new_db = FAISS.load_local("faiss_index", embeddings)
    # docs = new_db.similarity_search(user_question)
    # chain = get_conversational_chain()
    # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("Diplomaly Q&A Chatbot")

    if not st.session_state.vectorstore_created:

        #for now we dont pass in anything here;

        text = get_pdf_text()
        create_vector_store(text)

    if st.session_state.history_aware_retriever is None:
        create_retriever_with_history()

    if st.session_state.conversational_rag_chain is None:
        get_conversational_chain(st.session_state.history_aware_retriever)


    user_question = st.text_input("Ask any question about Unit 1", key="user_question")

    if user_question:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    # with st.sidebar:
    #     st.title("Menu:")
    #     pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
    #     if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
    #         with st.spinner("Processing..."):
    #             raw_text = get_pdf_text(pdf_docs)
    #             text_chunks = get_text_chunks(raw_text)
    #             get_vector_store(text_chunks, api_key)
    #             st.success("Done")

if __name__ == "__main__":
    main()
