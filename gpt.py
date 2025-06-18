from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

def load_model():
    """Load GPT-2 model and tokenizer"""
    print("Loading GPT-2 model... (this may take a moment on first run)")
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully!\n")
    return model, tokenizer

def predict_next_words(text, model, tokenizer, num_words=5):
    """Mode 1: Predict the next N words sequentially"""
    if not text.strip():
        print("Please enter some text!\n")
        return

    print(f"Input: '{text}'")
    print(f"Predicting next {num_words} words sequentially...")

    # Start with the input text
    current_text = text
    predicted_words = []

    # Generate word by word
    for _ in range(num_words):
        # Tokenize current text
        inputs = tokenizer.encode(current_text, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.logits[0, -1, :]  # Last token predictions

        # Get the most likely next token
        next_token_id = torch.argmax(predictions).item()
        next_word = tokenizer.decode(next_token_id)

        # Add to our sequence
        predicted_words.append(next_word)
        current_text += next_word

        # Stop if we hit an end token
        if next_token_id == tokenizer.eos_token_id:
            break

    # Clean up the predicted text
    predicted_text = "".join(predicted_words).strip()

    print(f"Predicted continuation: '{predicted_text}'")
    print(f"Full sequence: '{text} {predicted_text}'")
    print("-" * 60)

def predict_top_tokens(text, model, tokenizer, top_k=5):
    """Mode 2: Show top K predictions for the next single token with probabilities"""
    if not text.strip():
        print("Please enter some text!\n")
        return

    print(f"Input: '{text}'")
    print(f"Top {top_k} predictions for next token:")

    # Tokenize input
    inputs = tokenizer.encode(text, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.logits[0, -1, :]  # Last token predictions

    # Convert to probabilities
    probabilities = F.softmax(predictions, dim=0)

    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

    print("Rank | Token | Probability | Preview")
    print("-" * 45)

    for i, (prob, token_id) in enumerate(zip(top_k_probs, top_k_indices)):
        token = tokenizer.decode(token_id.item())
        # Create preview of what the full text would look like
        preview = f"{text}{token}"
        print(f"{i+1:4d} | '{token:8s}' | {prob.item():8.4f}")

    print("-" * 60)

def choose_mode():
    """Let user choose between the two modes"""
    print("\nChoose mode:")
    print("1. Sequential word prediction (predicts next N words in sequence)")
    print("2. Single token analysis (shows top 5 predictions with probabilities)")

    while True:
        try:
            choice = input("Enter mode (1 or 2): ").strip()
            if choice == "1":
                return "sequential"
            elif choice == "2":
                return "analysis"
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            return None

def get_word_count():
    """Get number of words to predict for sequential mode"""
    while True:
        try:
            count = input("How many words to predict? (default 5): ").strip()
            if not count:
                return 5
            count = int(count)
            if count > 0:
                return count
            else:
                print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            return 5

def main():
    """Main interactive loop"""
    print("=== Interactive GPT-2 Token Predictor ===")
    print("This demonstrates raw GPT-2 next-token prediction")
    print("Type 'quit' to exit, 'mode' to change modes\n")

    # Load the model
    model, tokenizer = load_model()

    # Initial mode selection
    current_mode = choose_mode()
    if current_mode is None:
        print("Goodbye!")
        return

    # Interactive loop
    while True:
        try:
            if current_mode == "sequential":
                print(f"\n[SEQUENTIAL MODE] - Predicting words in sequence")
            else:
                print(f"\n[ANALYSIS MODE] - Showing top token predictions")

            user_input = input("Enter text (or 'quit'/'mode'): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'mode':
                current_mode = choose_mode()
                if current_mode is None:
                    print("Goodbye!")
                    break
                continue

            if user_input:
                if current_mode == "sequential":
                    word_count = get_word_count()
                    predict_next_words(user_input, model, tokenizer, num_words=word_count)
                else:
                    predict_top_tokens(user_input, model, tokenizer, top_k=5)
            else:
                print("Please enter some text!\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()