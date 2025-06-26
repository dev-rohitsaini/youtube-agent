import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("OPEN_AI_KEY")
if key:
    os.environ["OPENAI_API_KEY"] = key

os.environ["OPENAI_MODEL"] = "gpt-4"



openai_key = os.getenv("OPEN_AI_KEY")

llm = LLM(model="gpt-4", api_key=os.getenv("OPEN_AI_KEY"))