#!/usr/bin/env python3
"""
OpenAI ChatGPT API Client Script
Requires: pip install openai python-dotenv
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChatGPTClient:
    def __init__(self, api_key=None):
        """
        Initialize the ChatGPT client

        Args:
            api_key (str, optional): OpenAI API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = []

    def chat(self, message, model="gpt-4", temperature=0.7, max_tokens=None, system_prompt=None):
        """
        Send a message to ChatGPT and get a response

        Args:
            message (str): The user message
            model (str): The model to use (default: gpt-4)
            temperature (float): Controls randomness (0.0 to 2.0)
            max_tokens (int, optional): Maximum tokens in response
            system_prompt (str, optional): System prompt to set behavior

        Returns:
            str: The assistant's response
        """
        try:
            # Prepare messages
            messages = []

            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history
            messages.extend(self.conversation_history)

            # Add current user message
            messages.append({"role": "user", "content": message})

            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            assistant_message = response.choices[0].message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message

        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def save_conversation(self, filename):
        """Save conversation history to a JSON file"""
        full_filename = os.path.join('chats', filename)
        with open(full_filename, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

    def load_conversation(self, filename):
        """Load conversation history from a JSON file"""
        with open(filename, 'r') as f:
            self.conversation_history = json.load(f)

def main():
    """Example usage of the ChatGPT client"""
    try:
        # Initialize the client
        client = ChatGPTClient()

        print("ChatGPT CLI Client")
        print("Type 'quit' to exit, 'clear' to clear history, 'save <filename>' to save conversation")
        print("-" * 50)

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                client.clear_history()
                print("Conversation history cleared.")
                continue
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                if filename:
                    client.save_conversation(filename)
                    print(f"Conversation saved to chats/{filename}")
                else:
                    print("Please provide a filename: save <filename>")
                continue
            elif not user_input:
                continue

            # Get response from ChatGPT
            response = client.chat(
                message=user_input,
                model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper responses
                temperature=0.7
            )

            print(f"\nChatGPT: {response}")

    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nTo set up your API key:")
        print("1. Create a .env file in the same directory as this script")
        print("2. Add the line: OPENAI_API_KEY=your_actual_api_key_here")
        print("3. Or set the environment variable: export OPENAI_API_KEY=your_actual_api_key_here")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()