import streamlit as st

from util.state import State
from util.generate import generate_cover_letter, split_resume
from util.job_search import search_postings
from ui.home import Home


def maybe_generate_cover_letter(update_button_state, toggle_generate_button, home_ui):
    """check if button pressed, state updated, and required fields are filled if so, generate new cover letter
    Args:
        update_button_state (State): the state to check if the button has been pressed
        *args (tuple): the state to check if the state has changed
    Returns:
    """

    toggle_generate_button(True)
    # getting the values from the text inputs and checking if they have changed
    cl_job = st.session_state["cl_job"]
    cl_company = st.session_state["cl_company"]
    cl_resume = st.session_state["cl_resume"]

    state_changed = update_button_state.check_state_changed(cl_job, cl_company, cl_resume)
    # job_title or company is empty, warn user
    if cl_job == "" or cl_company == "" or cl_resume is None:
        warning_message = "Please enter the following: "
        warning_message += "job title, " if cl_job == "" else ""
        warning_message += "company, " if cl_company == "" else ""
        warning_message += "resume" if cl_resume is None else ""
        st.warning(warning_message)
        home_ui.update_paper_container()
        toggle_generate_button(False)

    # job_title and company are filled, and state has changed, generate new cover letter
    elif cl_job and cl_company and state_changed:
        st.success("Generating cover letter")
        generate_cover_letter(st, home_ui)
        toggle_generate_button(False)

    # job_title and company are filled, but state has not changed, do nothing
    else:
        home_ui.update_paper_container()
        toggle_generate_button(False)


# initializing the streamlit UI
home_ui = Home(st)


# adding the job title and company to col1
with home_ui.col1:
    # inputs for job title and company
    st.text_input(
        "What job title do you want to write a cover letter for?", key="cl_job"
    )
    st.text_input("What company should the cover letter be for?", key="cl_company")
    help_text = """
        This is used to help openai generate an accurate cover letter base on your skills and experience\n
        We will not store your resume or any other personal information
    """
    
    # job posting search
    if "job_posting_search" not in st.session_state:
        st.session_state["job_posting_search"] = ""
    if "job_posting_search_results" not in st.session_state:
        st.session_state["job_posting_search_result"] = ""
    st.text_input("Add an optional job posting from Linkedin to further tailor your cover letter", value="", key="job_posting_search", on_change=search_postings, args=(st, ))
    file = st.file_uploader(
        label="upload your resume here",
        accept_multiple_files=False,
        help=help_text,
        key="cl_resume",
        on_change=split_resume,
        args=(st, ),
    )

# creating a state to check if the values for the cover letter have changed
update_button_state = State()
# attempts to generate cover letter on 'Generate' button press
if "generate_button_disabled" not in st.session_state:
    st.session_state["generate_button_disabled"] = False
def disable_generate_button(disable):
    st.session_state["generate_button_disabled"] = disable

# TODO: add a loading spinner while the cover letter is being generated
# TODO: add logic to replace the generate button with a regenerate button when no changes have been made to inputs
st.button(
    "GENERATE",
    on_click=maybe_generate_cover_letter,
    key="generate_button",
    args=(update_button_state, disable_generate_button, home_ui, ),
    disabled=st.session_state["generate_button_disabled"],
)
