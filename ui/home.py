import streamlit as st

class Home:
    def __init__(self):
        """Initializes the UI (markdown and formatting) for the home page
        
        Args:
        Returns:
        """
        
        st.title("📝🦜🔗CoverChain")
        st.markdown(
            """
            <style>
            .stApp {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            .st-bw {
                width: 100%;
                max-width: 600px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="stApp">', unsafe_allow_html=True)
        self.col1, self.col2 = st.columns([2, 1])
        
        # Display a constant prompt in col2
        with self.col2:
            self.prompt_container = st.empty() 
            self.update_paper_container()

    def update_paper_container(self, updated_text=None):
        """Updates the paper container with the given text
        
        Args:
            container (streamlit.container): The container to update
            text (str): The text to update the container with
        Returns:
        """
        
        if 'paper_container' not in st.session_state:
            st.session_state['paper_container'] = "fill out the form then press GENERATE to start getting results"
        if updated_text:
            st.session_state['paper_container'] = updated_text
        self.prompt_container.empty()
        self.prompt_container.markdown(
            f"""
            <div style="padding: 10px; background-color: #f9f9f9; border-radius: 5px; width: 250%;">
            <p style="font-family: Arial, sans-serif;">
            {st.session_state['paper_container']}
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

