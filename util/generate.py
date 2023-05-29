from util.file_loader import text_from_file, FileStatus

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory

def generate_embeddings_from_resume(st):
    """on file upload, get the text from the resume and store it in the session state
    Args:
    Returns:
    """
    
    if "cl_resume_retriever" not in st.session_state:
        st.session_state["cl_resume_retriever"] = None
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
    # if file is None, this means the file was removed, do cleanup
    else:
        st.session_state["cl_resume_retriever"] = None


def generate_cover_letter(st, home_ui):
    cl_job = st.session_state["cl_job"]
    cl_company = st.session_state["cl_company"]
    
    # creating a prompt
    # TODO: move propmts to a separate file
    cl_job_template = PromptTemplate(
        input_variables=["cl_job", "cl_company", "lookup_resume"],
        template="""
                    % QUERY:
                    write me a cover letter for the {cl_job} at {cl_company}
                    keep it between 250 and 400 words
                    {lookup_resume}
                    
                    % RESPONSE:
                """,
    )
    # getting retriever and memory for openai chat
    # only do so if the retriever exists
    retriever = st.session_state["cl_resume_retriever"]
        
    st.session_state["memory"] = memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # creating llm qa chain
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
    my_qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, condense_question_prompt=cl_job_template, verbose=True, memory=memory
    )
    # generating the cover letter, if not retriever, leave blank
    lookup_resume = """
        use the provided resume to generate the cover letter
        use the my name from the resume
        try to use skills and experiences from the resume that would best match the job if you know the company
    """
    question = cl_job_template.format(
        cl_job=cl_job, cl_company=cl_company, lookup_resume=lookup_resume
    )
    result = my_qa({"question": question})
    home_ui.update_paper_container(updated_text=result["answer"])