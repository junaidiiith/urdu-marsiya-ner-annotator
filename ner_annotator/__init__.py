from dotenv import load_dotenv
import os


load_dotenv('.env')  # Load environment variables from .env filecle

print("API Key: ", os.environ.get("OPENAI_API_KEY", "Not Set"))
