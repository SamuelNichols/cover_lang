import streamlit as st

@st.cache_resource
class State:
    def __init__(self):
        """simple state manager for to check if the state has changed
        
        Args:
        Returns:
        """
        self.prev_state = ()
        
    def check_state_changed(self, *args):
        """checks if the state has changed

        Args:
            *args (tuple): the state to checl
        Returns:
            bool: True if the state has changed, False otherwise
        """

        if self.prev_state != tuple(args):
            self.prev_state = args
            return True
        return False