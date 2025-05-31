import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph
import json
import re
import os

def parse_function_call(output: str):
    # Define regex pattern to capture function name and JSON payload
    match = re.search(r'<function[=./(]([a-zA-Z0-9_]+)[)>]?({.*})', output)
    
    if match:
        func_name = match.group(1)
        json_str = match.group(2)
        
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            payload = {}
        
        return func_name, payload
    
    return None, {}

async def predict(message, history, state):
    config = state
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = await graph.ainvoke({
        "messages": history_langchain_format
    }, config=config)

    output = gpt_response['messages'][-1].content
    if output.startswith("<function"):
        func_name, payload = parse_function_call(output)
        if func_name == "run_shell_command":
            cmd = payload.get("command")
            # e.g., execute cmd
            return f"<Executing '{cmd}' here>"
    return output

def update_key(key,state):
    state["configurable"] = {
        "api_key": key
    }
    gr.Info("API Key Configured...")
    return state

def load_key():
    try:
        with open(".groq_api_key", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except Exception:
        pass
    return ""

def set_groq_env_key():
    try:
        with open(".groq_api_key", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    os.environ["GROQ_API_KEY"] = line
                    break
    except Exception:
        pass

set_groq_env_key()

with gr.Blocks(theme=gr.themes.Soft()) as chat:
    state = gr.State()
    state.value = { }

    gr.Markdown("""
    # PCBot - Chat with Your Laptop
    Interact with your OS using natural language. Ask questions about files, processes, and system information.
    """)

    # --- API Key input always visible at the top ---
    with gr.Row():
        key = gr.Textbox(
            lines=1, 
            label="GROQ API Key",
            placeholder="Enter your API key here...",
            type="password",
            value=load_key()
        )
        button = gr.Button("Save Key", variant="primary")
    gr.Markdown("*Your API key is stored in .groq_api_key and will be loaded automatically if present.*")
    button.click(update_key, [key, state], [state])

    with gr.Tab("Chat"):
        chatbot = gr.ChatInterface(
            fn=predict, 
            type="messages",
            additional_inputs=[state],
            concurrency_limit=10,
            title="Chat Interface"
        )

    gr.Markdown("---\n*Powered by GROQ LLM*")


if __name__ == "__main__":
    chat.launch(server_port=7860)