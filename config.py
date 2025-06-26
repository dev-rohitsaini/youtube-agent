import os
from dotenv import load_dotenv
from crewai import LLM

load_dotenv()

openai_key = os.getenv("OPEN_AI_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
os.environ["OPENAI_MODEL"] = "gpt-4"

llm = LLM(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY")) 