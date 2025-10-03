def openai_response_text(openai_resp):
    return openai_resp["choices"][0]["message"]["content"]

# Function to extract response text for Ollama
def ollama_response_text(openai_resp):
    return openai_resp