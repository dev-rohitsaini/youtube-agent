from crewai.tools.base_tool import BaseTool
from tools.youtube_transcriber_tool import youtube_transcribe_tool
from crewai_tools.tools import YoutubeChannelSearchTool

# Create the YouTube channel search tool
yt_tool = YoutubeChannelSearchTool(youtube_channel_handle="@rajshamani")

class YouTubeTranscriberTool(BaseTool):
    name: str = "youtube_transcriber_tool"
    description: str = "Transcribes a YouTube video from the provided URL and returns the transcript."

    def _run(self, youtube_url: str) -> str:
        return youtube_transcribe_tool(youtube_url)

# Create an instance of the tool
youtube_transcriber_tool = YouTubeTranscriberTool(
    name="youtube_transcriber_tool",
    description="Transcribes a YouTube video from the provided URL and returns the transcript."
)