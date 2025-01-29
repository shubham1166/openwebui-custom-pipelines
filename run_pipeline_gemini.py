from gemini.gemini_manifold_pipeline import Pipeline

def main():
    # 1. Instantiate the pipeline
    pipeline = Pipeline()

    # 2. Prepare our input data
    user_message = "Hello, how are you?"
    model_id = "google/gemini-1.5-flash"  
    messages = [{"role": "user", "content": user_message}]
    
    body = {
        "user": {"email": "test@example.com"},
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
    }

    # 3. Call the pipe method
    response = pipeline.pipe(user_message, model_id, messages, body)

    # 4. Print the output
    if hasattr(response, "__iter__") and not isinstance(response, dict):
        # This check helps us handle streaming generators vs. JSON dictionaries.
        print("Streaming response:")
        for chunk in response:
            # The pipeline returns lines or chunked text
            # so we print without extra newlines.
            print(chunk, end="", flush=True)
        print()  # final newline
    else:
        # If it's a dict, it's likely the full JSON (non-stream).
        print("Full JSON response:")
        print(response)

if __name__ == "__main__":
    main()
