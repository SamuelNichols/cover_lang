import re
from typing import List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs

LINKEDIN_POSTING_CLASS = "show-more-less-html__markup show-more-less-html__markup--clamp-after-5 relative overflow-hidden"


def maybe_search_linkedin(url: str) -> str | None:
    """ensures that the url is a valid linkedin job search url"""
    search_pattern = r"^https?://www\.linkedin\.com/jobs/search\?"
    posting_pattern = r"^https?://www\.linkedin\.com/jobs/view/.*"
    # Check if the URL matches the expected pattern, if so get the job posting
    if re.match(search_pattern, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        current_job_id = query_params.get('currentJobId', [''])[0]
        job_posing_url = f"https://www.linkedin.com/jobs/view/{current_job_id}"
        return search_linkedin(job_posing_url)
    elif re.match(posting_pattern, url):
        return search_linkedin(url)
    return None


def search_linkedin(url: str):
    """Searches for job postings based on the query and returns a list of job postings"""
    # getting the html from the url
    html = requests.get(url)
    soup = BeautifulSoup(html.text, "html.parser")
    # find div with class LINKEDIN_POSTING_CLASS
    actual_post_content = soup.find("div", ["show-more-less-html__markup"])
    return actual_post_content

# TODO: add more job search websites
def search_postings(st):
    """Searches for job postings based on the url from job_posting_search and returns a list of job postings"""
    if "job_posting_search" not in st.session_state:
        st.session_state["job_posting_search"] = None
    
    job_posting_search = st.session_state["job_posting_search"]
    if job_posting_search != "":
        res = maybe_search_linkedin(job_posting_search)
        if res:
            st.session_state["job_posting_search_result"] = res
        else:
            st.warning("Please enter a valid linkedin job posting url")

