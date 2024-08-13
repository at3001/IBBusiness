import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import nest_asyncio
from llama_parse import LlamaParse

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec
import torch

from openai import OpenAI

st.session_state.client = OpenAI(api_key = api_key)


# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'history_aware_retriever' not in st.session_state:
    st.session_state.history_aware_retriever = None

if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'contextualize_q_chain' not in st.session_state:
    st.session_state.contextualize_q_chain = None

if 'valid_q_chain' not in st.session_state:
    st.session_state.valid_q_chain = None

def get_pdf_text():
    nest_asyncio.apply()
    parser = LlamaParse(
        api_key="llx-OmxZDgHY0WOPAYcxtKPD33KPQ56KPys4MGeYyVIlca2tqs4v",
        result_type="markdown",
        num_workers=4,
        verbose=True,
        language="en",
    )
    documents = parser.load_data("/Users/akshat/Downloads/ocr_unit1.pdf")
    text = '\n\n'.join([d.text for d in documents])
    return text

def create_vector_store(text):
    st.session_state.pc = Pinecone(api_key=pinecone_api_key)

    index_name = "langchain-test-index"  # change if desired

    existing_indexes = [index_info["name"] for index_info in st.session_state.pc.list_indexes()]

    if index_name not in existing_indexes:
        st.session_state.pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not st.session_state.pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    st.session_state.index = st.session_state.pc.Index(index_name)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    documents = [Document(page_content=split) for split in splits]
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    vectorstore = PineconeVectorStore.from_documents(documents, embeddings, index_name = "langchain-test-index")

    st.session_state.vectorstore = vectorstore.as_retriever()

def create_retriever_with_history():
    st.session_state.llm = ChatOpenAI(model="gpt-4", openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0)
    
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

    st.session_state.contextualize_q_chain = (contextualize_q_prompt | st.session_state.llm).with_config(
        tags=["contextualize_q_chain"]
    )

    st.session_state.history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, st.session_state.vectorstore, contextualize_q_prompt
    )
    # st.session_state.history_aware_retriever.include_metadata = True  # Include metadata

def get_conversational_chain():
    system_prompt = (
        "You are an assistant for question-answering tasks related to IB Business. "
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
    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)
    rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def validate_q_doc(question, docs):

    #Need to error proof this prompt using some kind of parser or validator
    prompt = """
    You are given a 'question' and a set of 'documents' that the question is support to be answered from.
    You are supposed to figure out whether the question can be answered based on the given documents.

    -----------
    Question: {question}

    -----------
    Answer:
    {documents}
    -----------

    If the question can be answered using information from the documents, respond with a 'yes', otherwise respond with a 'no'.
    Do not respond with anything but a 'yes' or 'no'
    """.format(question = question, documents = '\n'.join([d for d in docs]))

        
    response = st.session_state.client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt = prompt,
        max_tokens = 50,
        temperature = 0
    )

    response = response.choices[0].text.strip().lower()

    print(f"response: {response}")

    if response == 'yes':
        return True
    else:
        return False

    # We use this to validate if the given documents are appropriate to answer the question or not

def check_security_input():
    #Using ShieldGemma to validate input
    pass

def check_security_output():
    #Using ShieldGemma to validate output
    pass


def main():
    st.set_page_config(page_title="Document Genie", layout="wide")
    st.header("Document Genie Q&A Chatbot")

    # Initialize all components if they haven't been created yet
    if st.session_state.vectorstore is None:
        with st.spinner("Initializing the chatbot..."):
            text = get_pdf_text()
            create_vector_store(text)
            create_retriever_with_history()
            get_conversational_chain()
        st.success("Chatbot initialized successfully!")

    # Main chat interface
    st.subheader("Chat with Document Genie")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about the documents")

    #Reframed question
    reframed_question = st.session_state.contextualize_q_chain.invoke({"input": user_question, "chat_history": [m["content"] for m in st.session_state.chat_history]})
    # print("reframed question " + str(reframed_question))

    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Generate response
        with st.spinner("Thinking..."):
            response = st.session_state.conversational_rag_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": "abc123"}}
            )
            answer = response["answer"]

            retrieved_docs = [d.page_content for d in response.get("context", [])]

            print(f"retrieved docs {retrieved_docs}")


            #Check validity of the responses

            q_valid =  validate_q_doc(user_question, retrieved_docs)

            if not q_valid:
                answer = "Sorry, your question could not be answered based on our source material."



            

            

        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main()
