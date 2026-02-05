import os
import tempfile
import streamlit as st
from tempfile import NamedTemporaryFile
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore, FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool

from st_copy_to_clipboard import st_copy_to_clipboard

from components.prompt import (
    GeneratorPrompt,
    LinkedInMessagePrompt,
    ResumeModificationPrompt
)
from components.document import PDFDocumentLoader
from components.embedding import HuggingFaceEmbedding, GeminiEmbedding
from components.llm import CohereLLMChain, GeminiLLMChain


class CareerBuddyResponse(BaseModel):
    retrieved_content: str

option_generators = {
    "LinkedIn Message": LinkedInMessagePrompt(),
    "Cover Letter": GeneratorPrompt(),
    "Modify Resume": ResumeModificationPrompt()
}

os.environ["GOOGLE_API_KEY"] = st.secrets["gemini-api-key"]

def cover_me_app():
    st.title(":rainbow[Cover Me]")
    st.markdown(
        ":gray[**Do you want to stand out while applying for jobs? We got you covered! " \
        "Cover Me generates contextual cover letters based on your resume and the job description in seconds. We "
        "have added capability to rephrase your resume as well do try it out. Note: We're in the early " \
        "stages of development. Please bear with us and help us improve with your feedback. "
        "Share your suggestions and feature requests [here](https://forms.gle/UPXJBZxdiZy81XVQ9).**]", width="stretch"
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF)", type=["pdf"]
    )

    job_description = st.text_area(
        "Job Description",
        max_chars=5000,
        placeholder="Paste job description here",
        label_visibility="collapsed"
    )

    generator_option = st.selectbox(
        "Select generation type",
        list(option_generators.keys())
    )

    get_button = st.button("Generate")

    agent_executor = None
    vector_store = None

    if uploaded_file:
        with st.spinner("Indexing resume..."):
            vector_store = generate_index(uploaded_file)
            agent_executor = initiate_rag_agent(
                vector_store=vector_store,
                option_key=generator_option
            )

    if get_button:
        if not uploaded_file:
            st.error("Please upload a resume first.")
            return

        if not job_description.strip():
            st.error("Please paste a job description.")
            return

        with st.spinner("Generating content..."):
            try:
                result = agent_executor.invoke({
                    "messages": [{"role": "user", 
                    "content": "Use the job description provided here for the generation: " + job_description}]
                })
                output = result
            except Exception as e:
                st.error(f"Generation failed: {e}")
                output = None

        if output:
            st.success("Generated successfully")
            st.write(output["messages"][-1].content)
            st_copy_to_clipboard(output["messages"][-1].content)


@st.cache_resource
def generate_index(uploaded_file):
    temp_file = NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    resume_loader = PDFDocumentLoader(temp_file.name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = resume_loader.load_and_split(splitter)
    embedding = GeminiEmbedding().load()

    persist_path = tempfile.gettempdir() + "/resume_index.parquet"
    vector_store = SKLearnVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        persist_path=persist_path,
        serializer="parquet"
    )

    os.unlink(temp_file.name)
    return vector_store



def initiate_rag_agent(vector_store, option_key):

    llm = choose_llm().get_llm()

    raw_prompt = option_generators[option_key].get_template()
    template_str = getattr(raw_prompt, "template", str(raw_prompt))
    CAREER_SYSTEM_PROMPT_TEMPLATE = """
    You are an expert career assistant.
    Use the uploaded resume when needed.
    Be concise and provide actionable, prioritized suggestions.

    {template_str}

    Job Description:
    {input}

    When you reference resume content, include source metadata and keep answers focused.
    """
    prompt = CAREER_SYSTEM_PROMPT_TEMPLATE.format(template_str=template_str, input="{input}")    

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve top-2 relevant resume content."""
        retrieved_docs = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 5,
                "lambda_mult": 0.5
                }
                ).invoke(query)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    agent = create_agent(
        llm,
        tools=[retrieve_context],
        system_prompt=prompt
    )

    return agent

def choose_llm():
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["google-api-key"]
        st.info("Using Gemini LLM")
        return GeminiLLMChain()
    except Exception:
        os.environ["COHERE_API_KEY"] = st.secrets["cohere-api-key"]
        return CohereLLMChain()

def main():
    st.set_page_config(page_title="Cover Me", layout="centered")
    cover_me_app()


if __name__ == "__main__":
    main()
