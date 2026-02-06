import streamlit as st

home = st.Page("pages/home.py", title="Home")
career_buddy = st.Page("pages/career-buddy.py", title="Career Buddy")
cover_me = st.Page("pages/cover-me.py", title="Cover Me")

pg = st.navigation(
        {
            "Home": [home],
            "HR Apps": [career_buddy, cover_me]
        }
    )
st.set_page_config(page_title="Home", page_icon=":material/edit:")
pg.run()