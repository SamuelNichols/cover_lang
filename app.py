# import deps
import streamlit as st
from ui import HomeUI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


# initializing the streamlit UI
home_ui = HomeUI()


# adding the job title and company to col1
with home_ui.col1:
    cl_job = st.text_input('What job title do you want to write a cover letter for?')
    cl_company = st.text_input('What company should the cover letter be for?')
    
# Display a constant prompt in col2
with home_ui.col2:
    prompt_container = st.empty() 
    home_ui.update_paper_container(prompt_container, "")
    
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

# show response
if cl_job and cl_company:
    response = cl_chain.run({'cl_job': cl_job, 'cl_company': cl_company})
    home_ui.update_paper_container(prompt_container, response)
