from crewai import Task
from agents.research_analyst import research_analyst_agent
from agents.video_analyzer import video_analyzer_agent

# take environment variables from .env.

video_url = 'https://www.youtube.com/watch?v=MdeQMVBuGgY'

# Task 1: Transcription
transcribe_task = Task(
    description=f"Transcribe the YouTube video at this URL: {video_url}.",
    expected_output="A clear, readable content of the entire video in text format for entire video.",
    agent=video_analyzer_agent,
)

# Task 2: Analyze Transcript
analyze_task = Task(
    description="Analyze the transcription and extract key findings, frameworks, and summary insights.",
    expected_output="A structured  planning report including goals, steps, and key advice.",
    agent=research_analyst_agent,
    context=[transcribe_task],  # 🔗 Chain previous task output here
    
)

