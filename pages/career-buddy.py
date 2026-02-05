import streamlit as st
from components import agent


st.title("Career Buddy")

with st.spinner("Loading Career Buddy..."):
    graph = agent.build_graph()

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
# resume_text = st.text_area("Paste your resume here:", key="resume", height=300)
job_description = st.text_area("Paste the job description here:", key="job_description", height=300)

submit_button = st.button("Evaluate Fit")

if submit_button:
    input_state = {
        "resume_text": uploaded_file.read().decode("utf-8") if uploaded_file else resume_text, 
        "job_description": job_description
    }

    with st.spinner("Analyzing fit..."):
        response = graph.invoke(input_state)

    st.success("Analysis complete!")
    st.info("Fit Score:", response["final_score"])
    st.info("Explanation:", response["explanation"])
    st.info("Suggestions to Improve:", response["suggestion"])