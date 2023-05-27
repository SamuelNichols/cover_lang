import streamlit as st

class HomeUI:
    def __init__(self):
        """
        Initializes the UI (markdown and formatting) for the home page
        
        Parameters:
        
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

    def update_paper_container(self, container: st.container, text: str):
        """
        Updates the paper container with the given text
        
        Parameters:
            container (streamlit.container): The container to update
            text (str): The text to update the container with
        Returns:
        """
        container.empty()
        container.markdown(
            f"""
            <div style="padding: 10px; background-color: #f9f9f9; border-radius: 5px; width: 250%;">
            <p style="font-family: Arial, sans-serif;">
            {text}
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

