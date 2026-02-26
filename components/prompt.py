"""Prompt templates for generators.

Note: For LangChain 2.0 we return plain template strings and let the caller
wrap them in a PromptTemplate with the expected input variables.
"""


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
    - Give importance to global brand like JpMorgan and experience if present.
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
        self.prompt = self.generator_prompt

    def get_template(self):
        return self.prompt


class LinkedInMessagePrompt(Prompt):
    generator_prompt = """
        Create a linkedin chat message in the Format given below. Use the given 2 inputs below as variable to add context to the format.
        Format structure of cover letter:
        
        Dear <Name>,
        
        Instruction to write Body of message:
        
        - Write a message body that sounds professional, natural & written by human
        - body of message in the range of 50 to 250 words.
        - Give importance to global experience if available
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
        self.prompt = self.generator_prompt

    def get_template(self):
        return self.prompt

class ResumeModificationPrompt(Prompt):
    generator_prompt = """
                
        Objective: Modify the provided resume to align with the specified job description while ensuring that all facts in the resume are true. Focus on rephrasing and restructuring the content to highlight relevant skills and experiences.
        Job Description: {input}
        Resume: {context}
        Instructions:
        Review the Job Description:
        Carefully read through the job description to understand the key responsibilities, required skills, and qualifications.
        Take note of specific keywords or phrases that are frequently mentioned, as these are likely important to the employer.
        Examine the Resume:
        Read through the existing resume thoroughly.
        Identify experiences, skills, and accomplishments that are relevant to the job description.
        Highlight or make notes of sections that can be improved or rephrased to better match the job requirements.
        Modify the Resume:
        Begin rephrasing sections of the resume to incorporate keywords and phrases from the job description.
        Focus on aligning your past experiences with the responsibilities outlined in the job description. For example:
        If the job requires "project management skills," ensure that any relevant experience with project management is clearly articulated.
        Use action verbs and quantifiable achievements to enhance descriptions (e.g., "Led a team of 5" instead of "Responsible for managing a team").
        Adjust the order of experiences or sections if necessary to prioritize qualifications that are most relevant to the job.
        Ensure that any new phrasing accurately reflects your actual experiences and skills without exaggeration or misrepresentation.
        Maintain Professionalism:
        Keep the tone professional, clear, and concise throughout the resume.
        Ensure that formatting is consistent (e.g., font size, bullet points) and visually appealing.
        Avoid jargon or overly complex language; clarity is key.
        Final Review:
        After making modifications, review the updated resume for clarity, coherence, and flow.
        Check for grammatical errors or typos.
        Confirm that all information is accurate and effectively showcases how you meet the job requirements.
        Output:
        Provide a revised version of the resume with clear indications of changes made in response to the job description. Optionally, include a brief summary explaining how each modification aligns with specific elements of the job description.
        """
   
    def __init__(self):
        self.prompt = self.generator_prompt

    def get_template(self):
        return self.prompt
