from crewai import Agent
from utils.config import llm



research_analyst_agent = Agent(
    role="Research Analyst Agent",
    goal=(
        "Analyze research about the topic {topic} and provide valuable insights in strict JSON format. "
        "Follow this structure: {template}. "
        "Ensure all placeholders are replaced with real data. "
        "If data is missing, write 'Not available'."
    ),
    backstory="An AI expert specializing in research analysis, synthesizing information from multiple sources to provide actionable insights.",
    verbose=True,
    llm=llm,
    allow_delegation=True,
    tools=[]
)