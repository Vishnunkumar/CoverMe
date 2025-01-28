from langchain_core.prompts import ChatPromptTemplate


class Prompt:
    def get_template(self):
        pass


class GeneratorPrompt(Prompt):
    generator_prompt = """
    Create a cover letter in the Format given below. Use the given 2 inputs below as variable to add context to the format.
    Format structure of cover letter:
    
    Dear Hiring Team,
    
    Instruction to write Body of letter:
    
    - Write a cover letter body that sounds professional, natural & written by human
    - body of letter in the range of 150 to 250 words.
    - Body should be written based on steps 1 to 3 mentioned next:
    - Step 1: Extract important ATS key words from input 2 given below
    - Step 2: Extract details about the company from input 2 given below + internet
    - Step 3: Use extraction from step 1 & step 2 to combine with input 1 given below as context to write cover letter
    - Highlight the relevant experience from input 1 based keywords in step 1
    - Don’t mention terms like job description
    - Don’t write more than 2 lines on the company vision
    
    Best Regards
    
    Input 1 - Resume of applying candidate: {context}
    Input 2 - Job description for the job to apply: {input}
    Note: The output should only contain the cover letter as given in the format structure.
    Note 2: The last line of the cover letter should contain only the candidate's name Example: "John Doe"
    """

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(self.generator_prompt)

    def get_template(self):
        return self.prompt


class LinkedInMessagePrompt(Prompt):
    generator_prompt = """
        Create a linkedin chat message in the Format given below. Use the given 2 inputs below as variable to add context to the format.
        Format structure of cover letter:
        
        Dear <Name>,
        
        Instruction to write Body of message:
        
        - Write a message body that sounds professional, natural & written by human
        - body of message in the range of 50 to 150 words.
        - Body should be written based on steps 1 to 3 mentioned next:
        - Step 1: Extract important ATS key words from input 2 given below
        - Step 2: Extract details about the company from input 2 given below + internet
        - Step 3: Use extraction from step 1 & step 2 to combine with input 1 given below as context to write cover letter
        - Highlight the relevant experience from input 1 based keywords in step 1
        - Don’t mention terms like job description
        - Don’t write more than 1 lines on the company vision
        
        Best Regards
        
        Input 1 - Resume of applying candidate: {context}
        Input 2 - Job description for the job to apply: {input}
        Note: The output should only contain the linkedin message as given in the format structure.
        Note 2: The last line of the cover letter should contain only the candidate's name Example: "John Doe"
        """

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(self.generator_prompt)

    def get_template(self):
        return self.prompt

class ResumeModificationPrompt(Prompt):
    generator_prompt = """
        Modify the provided resume to align with the specified job description while ensuring that all facts in the resume are true. Focus on rephrasing and restructuring the content to highlight relevant skills and experiences.
        Job Description: {input}
        Resume: {context}
        Instructions:
        Review the job description and note key responsibilities and required skills.
        Examine the resume and identify relevant experiences and skills.
        Rephrase sections of the resume to incorporate keywords from the job description while keeping all information accurate.
        Adjust the order of experiences if necessary to prioritize relevant qualifications.
        Ensure the tone is professional and formatting is consistent.
        Output:
        Provide a revised version of the resume, highlighting changes made to align with the job description.
        """
   
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(self.generator_prompt)

    def get_template(self):
        return self.prompt
