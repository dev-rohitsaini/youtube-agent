from crewai import Agent
from tools.youtube_tools import  youtube_transcriber_tool
from config import llm
from tools.youtube_tools import yt_tool
from youtube_transcript_api import YouTubeTranscriptApi

# youtube_transcriber_tool
video_url = 'https://www.youtube.com/watch?v=MdeQMVBuGgY'

video_analyzer_agent = Agent(
    role="Video Analyzer Agent",
    goal="Analyze the video {video_url} and provide a summary, use whisper to transcribe the video. Provide all content of the video in text format",
    backstory="An AI expert at analyzing videos and providing important insights",
    verbose=True,
    llm=llm,
    allow_delegation=True,
    tools=[youtube_transcriber_tool],
)

video_id = "MdeQMVBuGgY"  # Extract this from the URL
transcript = YouTubeTranscriptApi.get_transcript(video_id)
# transcript is a list of dicts with 'text', 'start', 'duration'
full_text = " ".join([entry['text'] for entry in transcript])
print(full_text)
