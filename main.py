from crewai import Crew
from tasks.main_pipeline import  analyze_task

from agents.research_analyst import research_analyst_agent

crew = Crew(
    agents=[ research_analyst_agent],
    tasks=[ analyze_task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n📋 Final Output:\n")
    print(result)
