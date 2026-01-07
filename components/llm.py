from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMChain:
    def get_response(self, input_prompt):
        pass


class CohereLLMChain(LLMChain):
    def __init__(self):
        self.llm_chain = ChatCohere()

    def get_llm(self):
        return self.llm_chain

    def get_response(self, input_prompt):
        return self.get_llm().invoke({"input": input_prompt})

class GeminiLLMChain(LLMChain):
    def __init__(self):
        
        self.llm_chain = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            temperature=1.0, 
                            max_tokens=None,
                            timeout=None,
                            max_retries=2
                        )

    def get_llm(self):
        return self.llm_chain

    def get_response(self, input_prompt):
        return self.get_llm().invoke({"input": input_prompt})
