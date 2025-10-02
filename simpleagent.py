import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def load_and_set_api_keys():
    """
    Load API keys from .env file and ensure they're set in os.environ.

    Returns:
        dict: Dictionary with the API keys

    Raises:
        ValueError: If required API keys are not found
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    # Validate
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    # Ensure they're in os.environ (in case load_dotenv didn't set them)
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

    return {
        "openai_api_key": openai_api_key,
        "pinecone_api_key": pinecone_api_key
    }


def load_prompt(prompt_file="prompt.txt"):
    """
    Load the system prompt from a text file.

    Args:
        prompt_file: Path to the prompt file

    Returns:
        str: The prompt text

    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found")


def get_ai_response(user_input, system_prompt, model="gpt-4"):
    """
    Get a response from the OpenAI agent.

    Args:
        user_input: The user's question or input text
        system_prompt: The system prompt defining AI behavior
        model: The OpenAI model to use (default: gpt-4)

    Returns:
        str: The AI's response
    """
    # Initialize the chat model
    llm = ChatOpenAI(model=model, temperature=0.7)

    # Create messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    # Get response
    response = llm.invoke(messages)

    return response.content


def main():
    """
    Main function to run the interactive chat loop.
    """
    try:
        # Load API keys
        print("Loading API keys...")
        load_and_set_api_keys()

        # Load system prompt
        print("Loading system prompt...")
        system_prompt = load_prompt("prompt.txt")

        print("\nAI Agent initialized! Type 'exit' or 'quit' to end the conversation.\n")

        # Interactive loop
        while True:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            # Get and print AI response
            try:
                response = get_ai_response(user_input, system_prompt)
                print(f"\nAI: {response}\n")
            except Exception as e:
                print(f"Error getting response: {e}\n")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
