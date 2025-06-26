from crewai import Crew
from tasks.main_pipeline import transcribe_task, analyze_task
from agents.video_analyzer import video_analyzer_agent
from agents.research_analyst import research_analyst_agent

crew = Crew(
    agents=[video_analyzer_agent, research_analyst_agent],
    tasks=[transcribe_task, analyze_task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n📋 Final Output:\n")
    print(result)
