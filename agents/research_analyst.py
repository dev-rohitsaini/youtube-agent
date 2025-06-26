from crewai import Agent
from tools.youtube_transcriber_tool import youtube_transcriber_tool
from utils.config import llm

youtube_url = "https://www.youtube.com/watch?v=-bt_y4Loofg"

template = """
Video Title: {{video_title}}
Key Insights:
1. {{insight_1}}
2. {{insight_2}}
3. {{insight_3}}

Summary:
{{summary}}

Actionable Points:
- {{point_1}}
- {{point_2}}
"""

goal_text = (
    f"Analyze text from youtube_transcriber_tool for video URL {youtube_url} and provide valuable insights. "
    f"Follow this structure: {template}. "
    "Ensure all placeholders are replaced with real data. "
    "If data is missing, write 'Not available'."
)

research_analyst_agent = Agent(
    role="Research Analyst Agent",
    goal=goal_text,
    backstory="An AI expert specializing in research analysis, synthesizing information from multiple sources to provide actionable insights.",
    verbose=True,
    llm=llm,
    allow_delegation=True,
    tools=[youtube_transcriber_tool]
)
