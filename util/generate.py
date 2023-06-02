from util.file_loader import text_from_file, FileStatus

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

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


def split_job_postings(st, text):
    """on job posting url upload, get the text from the job posting and store it in the session state as split text"""
    st.session_state["job_posting_search"] = text
    maybe_invalidate_vectordb(st)
    resume_text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=20)
    split_doc = resume_text_splitter.split_text(text)
    st.session_state["job_posting_search_result_split"] = split_doc
    
def split_resume(st):
    """on file upload, get the text from the resume and store it in the session state as split text"""
    file = None
    if "cl_resume" in st.session_state:
        file = st.session_state["cl_resume"]
    if file is not None:
        maybe_invalidate_vectordb(st)
        text, status = text_from_file(file)
        if status != FileStatus.FILE_PARSED:
            st.warning("currently supported file types: pdf")
            return
        if num_tokens_from_string(text) < 4500:
            resume_text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=50)
            split_doc = resume_text_splitter.split_text(text)
            st.success("file uploaded successfully")
            st.session_state["cl_resume_split"] = split_doc
        else:
            st.warning("resume is too long, please upload a shorter resume")
    else:
        maybe_invalidate_vectordb(st)
        st.session_state["cl_resume_split"] = None
    
def maybe_invalidate_vectordb(st):
    if "vector_db" in st.session_state and st.session_state["vector_db"] is not None:
            vectordb = st.session_state["vector_db"]
            print("removing vectorstore")
        
def build_resume_retriever(st):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    resume_db = FAISS.from_texts(st.session_state["cl_resume_split"], embeddings)
    return resume_db.as_retriever()

def build_job_posting_retriever(st):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    if "job_posting_search_result_split" not in st.session_state or st.session_state["job_posting_search_result_split"] is None:
        return None
    job_posting_db = FAISS.from_texts(st.session_state["job_posting_search_result_split"], embeddings)
    return job_posting_db.as_retriever()    

def generate_cover_letter(st, home_ui):
    st.session_state["run_cost"] = 0
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
    resume_retriever = build_resume_retriever(st)
    job_posting_retriever = build_job_posting_retriever(st)
        
    st.session_state["memory"] = memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # creating llm qa chain
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
    my_qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=resume_retriever, condense_question_prompt=cl_job_template, verbose=True, memory=memory
    )
    # generating the cover letter, if not retriever, leave blank
    lookup_resume = """
        use skills and experiences from my resume to write the cover letter
        only get skills and experiences that are relevant to the job
        get the skills and experiences from the resume that are relevant to the job
        use the resume creators name for the cover letter
    """
    llm = OpenAI(temperature=0.9, model_name="text-davinci-003")
    job_posting_summary_template = """Create a list of the following, leave out if not found or unsure
    the summary can be up to 200 words
    - About the company
    - About the team
    - Required skills for the role (anything that sounds like a requirement)
    - Preferred skills for the role (anything that sounds like a preference)
    
    {text}
    
    CONCISE SUMMARY:
    """
    PROMPT = PromptTemplate(template=job_posting_summary_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    texts = st.session_state["job_posting_search_result_split"]
    docs = [Document(page_content=t) for t in texts[:3]]
    with get_openai_callback() as cb:
        result = chain({"input_documents": docs}, return_only_outputs=True)
        print("result: ", result)
        print("openai usage: ", cb)
        
   
    # about_company = job_posting_retriever.get_relevant_documents("information about the compmany mission and team")
    # for document in about_company:
    #     print("\n\n\nabout company: ", document)
    
    # about_job = job_posting_retriever.get_relevant_documents("requirements for the job from job posting")
    # for document in about_job:
    #     print("\n\n\nabout job: ", document)
    # question = cl_job_template.format(
    #     cl_job=cl_job, cl_company=cl_company, lookup_resume=lookup_resume
    # )
    # with get_openai_callback() as cb:
    #     result = my_qa({"question": question})
    #     print("openai usage: ", cb)
        
    # home_ui.update_paper_container(updated_text=result["answer"])
    home_ui.update_paper_container(updated_text="test")
    