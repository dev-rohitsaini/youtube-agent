from crewai import Agent
from tools.youtube_transcriber_tool import youtube_transcriber_tool
from config import llm


# youtube_transcriber_tool
video_url = 'https://www.youtube.com/watch?v=-bt_y4Loofg'

video_analyzer_agent = Agent(
    role="Video Analyzer Agent",
    goal="Analyze the video {video_url} and provide a concise summary. Transcribe the video in chunks and provide a summary of the content. The total tokens for both transcription and summary should not exceed 10,000 tokens. Please ensure each chunk is manageable in terms of length.",
    backstory="An AI expert at analyzing videos and providing important insights",
    verbose=True,
    llm=llm,
    allow_delegation=True,
    tools=[youtube_transcriber_tool],  # type: ignore
)
