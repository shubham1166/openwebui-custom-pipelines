from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_VERSION: str
        AZURE_OPENAI_MODELS: str
        AZURE_OPENAI_MODEL_NAMES: str

    def __init__(self):
        self.type = "manifold"
        self.name = "Azure OpenAI: "
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                "AZURE_OPENAI_MODELS": os.getenv("AZURE_OPENAI_MODELS", "gpt-35-turbo;gpt-4o"),
                "AZURE_OPENAI_MODEL_NAMES": os.getenv("AZURE_OPENAI_MODEL_NAMES", "GPT-35 Turbo;GPT-4o"),
            }
        )
        self.set_pipelines()

    def set_pipelines(self):
        models = self.valves.AZURE_OPENAI_MODELS.split(";")
        model_names = self.valves.AZURE_OPENAI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model, "name": name} for model, name in zip(models, model_names)
        ]
        print(f"azure_openai_manifold_pipeline - models: {self.pipelines}")

    async def on_valves_updated(self):
        self.set_pipelines()        

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        headers = {
            "api-key": self.valves.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        # URL for Chat Completions in Azure OpenAI
        url = (
            f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/"
            f"{model_id}/chat/completions?api-version={self.valves.AZURE_OPENAI_API_VERSION}"
        )

        # --- Define the allowed parameter sets ---
        # (1) Default allowed params (non-o1)
        allowed_params_default = {
            "messages",
            "temperature",
            "role",
            "content",
            "contentPart",
            "contentPartImage",
            "enhancements",
            "dataSources",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "function_call",
            "funcions",
            "tools",
            "tool_choice",
            "top_p",
            "log_probs",
            "top_logprobs",
            "response_format",
            "seed",
        }
        
        # (2) o1 models allowed params
        allowed_params_o1 = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "n",
            "stream",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user"
        }
        
        # Simple helper to detect if it's an o1 model
        # Adjust this check based on how your "o1" models are named
        def is_o1_model(m: str) -> bool:
            return "o1" in m or m.endswith("o")

        # Remap user field if needed
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"].get("id", str(body["user"]))

        # Pick which allowed params to filter by
        if is_o1_model(model_id):
            filtered_body = {k: v for k, v in body.items() if k in allowed_params_o1}
        else:
            filtered_body = {k: v for k, v in body.items() if k in allowed_params_default}
        
        # Log which fields were dropped
        if len(body) != len(filtered_body):
            dropped_keys = set(body.keys()) - set(filtered_body.keys())
            print(f"Dropped params: {', '.join(dropped_keys)}")

        try:
            r = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=True,
            )
            r.raise_for_status()
            if filtered_body.get("stream"):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            # If the request object exists, return its text
            if 'r' in locals() and r is not None:
                return f"Error: {e} ({r.text})"
            else:
                return f"Error: {e}"
