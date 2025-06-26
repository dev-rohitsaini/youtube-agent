from crewai import Task
from agents.research_analyst import research_analyst_agent


# take environment variables from .env.


# Task 2: Analyze Transcript
analyze_task = Task(
    description="Analyze the transcription and extract key findings, frameworks, and summary insights.",
    expected_output="A structured  planning report including goals, steps, and key advice.",
    agent=research_analyst_agent,
    context=[],  # 🔗 Chain previous task output here
    
)

