import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Initialize the client (it automatically look for the GROQ_API_KEY variable in .env folder)

client = Groq()

try:
    completion = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role' : "user",
                'content' : 'Give me one word answer for capital of India.'
            }
        ],
        temperature=0.1,
    )

    print("\n--- Response ---")
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"Error: {e}")
