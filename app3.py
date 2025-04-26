import os
import tempfile
from tempfile import NamedTemporaryFile

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import CharacterTextSplitter
from st_copy_to_clipboard import st_copy_to_clipboard


from components.document import PDFDocumentLoader
from components.embedding import HuggingFaceEmbedding
from components.llm import CohereLLMChain
from components.prompt import GeneratorPrompt, LinkedInMessagePrompt

option_generators = {
    "LinkedIn Message": LinkedInMessagePrompt(),
    "Cover Letter": GeneratorPrompt()
}


def cover_me_app():
    st.title("Cover Me")
    st.markdown(
        "**Do you want to stand out while applying for jobs? We got you covered! Cover Me generates contextual cover "
        "letters based on your resume and the job description in seconds.**")
    st.markdown(
        "Note: We're in the early stages of development. Please bear with us and help us improve with your feedback. "
        "Share your suggestions and feature requests [here](https://forms.gle/UPXJBZxdiZy81XVQ9).")
    st.divider()

    uploaded_file = st.file_uploader("Upload your CV/Resume here:", type=["pdf"], key="cv-upload")
    job_description = st.text_area("Job Description:",
                                   max_chars=5000,
                                   placeholder="Paste job description here",
                                   label_visibility="collapsed",
                                   key="job-desc")
    generator_option = st.selectbox(
        "Select the type of generation you would like to do:",
        ("LinkedIn Message", "Cover Letter")
    )
    get_button = st.button("Get generated content")

    if uploaded_file:
        with st.spinner("Indexing the resume"):
            retriever = generate_index(uploaded_file)
            llm_chain = initiate_llm(retreiver=retriever,
                                     option_key=generator_option)

    if get_button:
        cover_letter = llm_chain.invoke({"input": job_description})["answer"]
        st.info("Content generated successfully:")
        st.success(cover_letter)
        st_copy_to_clipboard(cover_letter)


@st.cache_resource()
def generate_index(uploaded_file):
    tf = NamedTemporaryFile(dir='./', suffix='.pdf', delete=False)
    tf.write(uploaded_file.getbuffer())
    resume_loader = PDFDocumentLoader(tf.name)
    resume_text_splitter = CharacterTextSplitter()
    resume_data = resume_loader.load_and_split(resume_text_splitter)
    embedding = HuggingFaceEmbedding().load()
    persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
    vectorstore = SKLearnVectorStore.from_documents(documents=resume_data, embedding=embedding,
                                                    persist_path=persist_path, serializer="parquet")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': len(resume_data)})
    tf.close()
    os.unlink(tf.name)
    return retriever


def initiate_llm(retreiver, option_key):
    os.environ["COHERE_API_KEY"] = st.secrets["cohere-api-key"]
    cohere_llm = CohereLLMChain().get_llm()
    prompt_template = option_generators[option_key].get_template()
    cohere_question_answer_chain = create_stuff_documents_chain(cohere_llm, prompt_template)
    cohere_rag_chain = create_retrieval_chain(retreiver, cohere_question_answer_chain)
    return cohere_rag_chain


def main():
    st.set_page_config()
    cover_me_app()


if __name__ == "__main__":
    main()
