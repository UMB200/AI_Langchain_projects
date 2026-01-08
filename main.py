from dotenv import load_dotenv
import os
load_dotenv()

def main():
    # print(os.getenv("OPENAI_API_KEY")) the same as os.environ.get("OPENAI_API_KEY") but in the course used os.environ.get("OPENAI_API_KEY")
    print(os.environ.get("OPENAI_API_KEY"))
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
