import os
import operator
from langchain.tools import tool
from typing import TypedDict, Dict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_perplexity import ChatPerplexity


llm = ChatPerplexity(temperature=0)

class FitState(TypedDict):
    resume_text: Annotated[str, operator.add]
    jd_text: Annotated[str, operator.add]

    resume_skills: Annotated[list[str], operator.add]
    jd_skills: Annotated[list[str], operator.add]
    experience_years: Annotated[int, operator.add]
    required_years: Annotated[int, operator.add]

    skill_score: Annotated[float, operator.add]
    experience_score: Annotated[float, operator.add]
    final_score: Annotated[float, operator.add]

    explanation: Annotated[str, operator.add]
    suggestion: Annotated[str, operator.add]

def llm_json(prompt: str) -> Dict:
    parser = JsonOutputParser()
    chain = ChatPromptTemplate.from_template(prompt) | llm | parser
    return chain.invoke({})

def llm_number(prompt: str) -> float:
    response = llm.invoke(prompt)
    return float(response.content.strip())

def extract_resume(state: FitState) -> FitState:
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured data from resumes."),
        ("human", """
            Extract STRICT JSON:
            {{
              "skills": ["..."],
              "years_experience": number
            }}
            
            Resume:
            {resume}
            """)
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "resume": state["resume_text"]
    })

    state["resume_skills"] = result.get("skills", [])
    state["experience_years"] = result.get("years_experience", 0)

    return state


def extract_jd(state: FitState) -> FitState:
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured data from job descriptions."),
        ("human", """
Extract STRICT JSON:
{{
  "required_skills": ["..."],
  "min_experience": number
}}

Job Description:
{jd}
""")
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "jd": state["jd_text"]
    })

    state["jd_skills"] = result.get("required_skills", [])
    state["required_years"] = result.get("min_experience", 0)

    return state


def skill_match(state: FitState) -> FitState:
    resume = set(map(str.lower, state["resume_skills"]))
    jd = set(map(str.lower, state["jd_skills"]))

    match_ratio = len(resume & jd) / max(len(jd), 1)
    state["skill_score"] = round(match_ratio * 60, 2)
    return state

def experience_match(state: FitState) -> FitState:
    ratio = min(state["experience_years"] / state["required_years"], 1.2)
    state["experience_score"] = round(ratio * 40, 2)
    return state

def aggregate_score(state: FitState) -> FitState:
    state["final_score"] = round(
        state["skill_score"] + state["experience_score"], 2
    )
    return state

def explanation_node(state: FitState) -> FitState:
    prompt = f"""
    Explain why the resume scored {state["final_score"]}%.

    Resume skills: {state["resume_skills"]}
    Job skills: {state["jd_skills"]}
    Experience: {state["experience_years"]} vs required {state["required_years"]}

    Be concise and specific.
    """
    state["explanation"] = llm.invoke(prompt).content
    return state

def suggestion_node(state: FitState) -> FitState:
    prompt = f"""
    Based on the explanation {state["explanation"]} provided by you, act as proper career expert and provide good and valid suggestions 
    """
    state["suggestion"] = llm.invoke(prompt).content
    return state

def build_graph():
    graph = StateGraph(FitState)

    graph.add_node("extract_resume", extract_resume)
    graph.add_node("extract_jd", extract_jd)

    graph.add_node("skill_match", skill_match)
    graph.add_node("experience_match", experience_match)

    graph.add_node("aggregate", aggregate_score)
    graph.add_node("explain", explanation_node)
    graph.add_node("suggestion", suggestion_node)

    graph.set_entry_point("extract_resume")
    graph.add_edge("extract_resume", "extract_jd")

    graph.add_edge("extract_jd", "skill_match")
    graph.add_edge("extract_jd", "experience_match")

    graph.add_edge("skill_match", "aggregate")
    graph.add_edge("experience_match", "aggregate")

    graph.add_edge("aggregate", "explain")
    graph.add_edge("explain", "suggestion")

    graph.add_edge("suggestion", END)

    return graph.compile()

# if __name__ == "__main__":
#     graph = build_graph()

#     input_state = {
#     "resume_text": """
#     Senior Backend Engineer with 7+ years of experience building Java-based
#     microservices. Strong expertise in Spring Boot, REST APIs, and SQL databases.
#     Worked in fintech and payments systems with high-volume transaction processing.
#     Experience deploying applications using Docker and basic exposure to AWS.
#     """,

#     "jd_text": """
#     We are hiring a Backend Engineer with 6+ years of experience.
#     Strong skills required in Java, Spring Boot, and Kafka.
#     Experience with distributed systems and cloud platforms like AWS is preferred.
#     Background in fintech or payments is a plus.
#     """
#     }


#     result = graph.invoke(input_state)

#     print("\n===== FINAL OUTPUT =====")
#     print("Score:", result["final_score"])
#     print("Explanation:", result["explanation"])
#     print("Suggestions:", result["suggestion"])
