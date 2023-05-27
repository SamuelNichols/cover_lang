# import deps
import time
import streamlit as st
from uitl.state import State
from ui.home import Home
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

def maybe_generate_cover_letter(update_button_state: State, *args):
    """check if button pressed, state updated, and required fields are filled if so, generate new cover letter
    Args:
        update_button_state (State): the state to check if the button has been pressed
        *args (tuple): the state to check if the state has changed
    Returns:
    """
    
    if st.button("Generate"):
        # getting the values from the text inputs and checking if they have changed
        cl_job = st.session_state['cl_job']
        cl_company = st.session_state['cl_company']
        cl_resume = st.session_state['cl_resume']
        print(cl_resume)
        state_changed = update_button_state.check_state_changed(cl_job, cl_company)
        
        if cl_job == "" or cl_company == "":
            st.warning("Please enter a job title and company")
            home_ui.update_paper_container()
        elif cl_job and cl_company and state_changed:
            # response = cl_chain.run({'cl_job': cl_job, 'cl_company': cl_company})
            result = f"Textbox updated at {time.time()}"
            home_ui.update_paper_container(updated_text=result)
        else:
            home_ui.update_paper_container()
            
def get_text_from_resume():
    file = st.session_state['cl_resume']
    if file:
        if file.type == "application/pdf":
                st.success("file uploaded successfully")
                # TODO: use pdf loader to get text from pdf
        else:
            st.warning("currently supported file types: pdf")

# initializing the streamlit UI
home_ui = Home()


# adding the job title and company to col1
with home_ui.col1:
    st.text_input('What job title do you want to write a cover letter for?', key="cl_job")
    st.text_input('What company should the cover letter be for?', key="cl_company")
    help_text = """
        This is used to help openai generate an accurate cover letter base on your skills and experience\n
        We will not store your resume or any other personal information
    """
    st.file_uploader(label="upload your resume here", accept_multiple_files=False, help=help_text, key="cl_resume", on_change=get_text_from_resume)
    
# prompt template
cl_job_template = PromptTemplate(
    input_variables=['cl_job', 'cl_company'],
    template="""
        % QUERY:
        write me a cover letter for the {cl_job} at {cl_company}
        
        % RESPONSE:
    """
)


# llms
llm=OpenAI(temperature=0.9, verbose=True)
cl_chain = LLMChain(llm=llm, prompt=cl_job_template, verbose=True)

# creating a state to check if the values for the cover letter have changed
update_button_state = State()
# attempts to generate cover letter on 'Generate' button press
maybe_generate_cover_letter(update_button_state)
