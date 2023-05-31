from util.file_loader import text_from_file, FileStatus

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import tiktoken

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# constants for cost calculation
ADA_V2_TEXT_EMBED_COST = 0.0004
ADA_V2_TEXT_EMBED_TOKENS = 1000

def embedding_cost_estimate(num_tokens:int, model_cost: str, model_tokens: str) -> int:
    """Returns the estimated cost of an openai text embedding run
    Args:
        num_tokens: number of tokens in the text
        model_cost: cost of the model per 1000 tokens
        model_tokens: number of tokens the model can process
    Returns:
        estimated cost of the model
    """
    # calculate the cost then round to 4 decimal places
    cost = (num_tokens/model_tokens) * model_cost
    return float("{:.4f}".format(cost))

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

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
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
            split_doc = text_splitter.split_text(text)
            num_tokens = num_tokens_from_string(text)
            if num_tokens_from_string(text) < 4500:
                st.success("file uploaded successfully")
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
                vectordb = Chroma.from_texts(split_doc, embeddings)
                print("embedding cost: ", embedding_cost_estimate(num_tokens, ADA_V2_TEXT_EMBED_COST, ADA_V2_TEXT_EMBED_TOKENS))
                retriever = vectordb.as_retriever(search_type="similarity")
                st.session_state["cl_resume_retriever"] = retriever
                st.session_state["resume_vector_db"] = vectordb
            else:
                st.warning("resume is too long, please upload a shorter resume")

        else:
            st.warning("currently supported file types: pdf")
            st.session_state["cl_resume_retriever"] = None
    # if file is None, this means the file was removed, do cleanup
    else:
        if "resume_vector_db" in st.session_state and st.session_state["resume_vector_db"] is not None:
            vectordb = st.session_state["resume_vector_db"]
            Chroma.delete_collection(vectordb)
        else:
            st.session_state["resume_vector_db"] = None
            
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
                    make it sound like a human wrote it, don't make it too formal
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
        use skills and experiences from my resume to write the cover letter
        only get skills and experiences that are relevant to the job
        use the resume creators name for the cover letter
    """
    question = cl_job_template.format(
        cl_job=cl_job, cl_company=cl_company, lookup_resume=lookup_resume
    )
    with get_openai_callback() as cb:
        result = my_qa({"question": question})
        print("openai usage: ", cb)
    home_ui.update_paper_container(updated_text=result["answer"])