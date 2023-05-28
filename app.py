import time
import streamlit as st

from util.state import State
from ui.home import Home
from util.file_loader import text_from_file, FileStatus

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def generate_embeddings_from_resume():
    """on file upload, get the text from the resume and store it in the session state
    Args:
    Returns:
    """

    file = st.session_state["cl_resume"]
    if file is not None:
        text, status = text_from_file(file)
        if status == FileStatus.FILE_PARSED:
            st.success("file uploaded successfully")
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
            split_doc = text_splitter.split_text(text)
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma.from_texts(split_doc, embeddings)
            retriever = vectordb.as_retriever(search_type="similarity")
            st.session_state["cl_resume_retriever"] = retriever

        else:
            st.warning("currently supported file types: pdf")
            st.session_state["cl_resume_retriever"] = None
    else:
        st.session_state["cl_resume_text"] = None


def generate_cover_letter(cl_job, cl_company):
    # creating a prompt
    cl_job_template = PromptTemplate(
        input_variables=["cl_job", "cl_company", "lookup_resume"],
        template="""
                    % QUERY:
                    write me a cover letter for the {cl_job} at {cl_company}
                    {lookup_resume}
                    
                    % RESPONSE:
                """,
    )
    # getting retriever and memory for openai chat
    retriever = st.session_state["cl_resume_retriever"]
    st.session_state["memory"] = memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # creating llm qa chain
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
    if retriever is not None:
        my_qa = ConversationalRetrievalChain.from_llm(
            llm, retriever, cl_job_template, verbose=True, memory=memory
        )
    else:
        my_qa = ConversationalRetrievalChain.from_llm(
            llm, cl_job_template, verbose=True, memory=memory
        )
    # generating the cover letter
    lookup_resume = (
        ""
        if retriever is None
        else "use the provided resume to generate the cover letter"
    )
    question = cl_job_template.format(
        cl_job=cl_job, cl_company=cl_company, lookup_resume=lookup_resume
    )
    result = my_qa({"question": question})
    home_ui.update_paper_container(updated_text=result['answer'])


def maybe_generate_cover_letter(update_button_state: State, *args):
    """check if button pressed, state updated, and required fields are filled if so, generate new cover letter
    Args:
        update_button_state (State): the state to check if the button has been pressed
        *args (tuple): the state to check if the state has changed
    Returns:
    """

    if st.button("Generate"):
        # getting the values from the text inputs and checking if they have changed
        cl_job = st.session_state["cl_job"]
        cl_company = st.session_state["cl_company"]

        state_changed = update_button_state.check_state_changed(cl_job, cl_company)
        # job_title or company is empty, warn user
        if cl_job == "" or cl_company == "":
            st.warning("Please enter a job title and company")
            home_ui.update_paper_container()

        # job_title and company are filled, and state has changed, generate new cover letter
        elif cl_job and cl_company and state_changed:
            st.success("Generating cover letter")
            generate_cover_letter(cl_job, cl_company)

        # job_title and company are filled, but state has not changed, do nothing
        else:
            home_ui.update_paper_container()


# initializing the streamlit UI
home_ui = Home(st)


# adding the job title and company to col1
with home_ui.col1:
    st.text_input(
        "What job title do you want to write a cover letter for?", key="cl_job"
    )
    st.text_input("What company should the cover letter be for?", key="cl_company")
    help_text = """
        This is used to help openai generate an accurate cover letter base on your skills and experience\n
        We will not store your resume or any other personal information
    """
    file = st.file_uploader(
        label="upload your resume here",
        accept_multiple_files=False,
        help=help_text,
        key="cl_resume",
        on_change=generate_embeddings_from_resume,
    )


# creating a state to check if the values for the cover letter have changed
update_button_state = State()
# attempts to generate cover letter on 'Generate' button press
maybe_generate_cover_letter(update_button_state)
