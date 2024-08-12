import os
import streamlit as st
from together import Together
import time

# Set page config at the very beginning
st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")

# Initialize the Together client
@st.cache_resource
def get_together_client():
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        st.error("API Key not found. Please set the TOGETHER_API_KEY environment variable.")
        st.stop()
    return Together(api_key=api_key)

# Function to generate response
def generate_response(prompt, model, temperature, max_tokens):
    try:
        client = get_together_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["\u0000"],  # Use Unicode null character instead of empty string
            stream=True
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        yield None

# Main app
def main():
    st.title("🤖 AI Chat Assistant")

    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header("Model Settings")
        model = st.selectbox(
            "Select Model",
            ["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"]
        )
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=1000, value=512, step=50)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a container to hold the chat messages
    chat_container = st.container()

    # Display chat messages from history on app rerun
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        with chat_container:
            st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # Process the streaming response
                for chunk in generate_response(prompt, model, temperature, max_tokens):
                    if chunk is not None:
                        full_response += chunk
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Display API key status
    with st.sidebar:
        if os.environ.get('TOGETHER_API_KEY'):
            st.success("API Key detected in environment variables.")
        else:
            st.error("API Key not found. Please set the TOGETHER_API_KEY environment variable.")

    # Add a button to clear chat history
    with st.sidebar:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            chat_container.empty()

if __name__ == "__main__":
    main()
