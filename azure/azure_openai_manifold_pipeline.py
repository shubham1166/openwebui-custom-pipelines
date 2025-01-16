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
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        headers = {
            "api-key": self.valves.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        # 1) Read the 'user' email from body
        # user_id = body.get("user", {})
        # user_name = user_id.get("email", "").split("@")[0]
        user_name = body.get("name", {})

        # 2) Build the base URL *manually* to preserve `@` in the `source`
        #    This ensures the server sees `source=you@company.com` literally
        #    instead of `source=you%40company.com`
        if user_name:
            full_url = (
                f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{model_id}/chat/completions"
                f"?api-version={self.valves.AZURE_OPENAI_API_VERSION}&source={user_name}"
            )
        else:
            # If we have no email, just omit the source from the query string
            full_url = (
                f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{model_id}/chat/completions"
                f"?api-version={self.valves.AZURE_OPENAI_API_VERSION}"
            )

        # --- Define the allowed parameter sets ---
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

        allowed_params_o1 = {
            "model",
            "messages",
            "top_p",
            "n",
            "max_completion_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",  # <--- still here too
        }

        def is_o1_model(m: str) -> bool:
            return "o1" in m or m.endswith("o")

        # If it's an o1 model, do a "fake streaming" approach
        if is_o1_model(model_id):
            body.pop("stream", None)  # only remove 'stream' if present
            filtered_body = {k: v for k, v in body.items() if k in allowed_params_o1}

            if len(body) != len(filtered_body):
                dropped_keys = set(body.keys()) - set(filtered_body.keys())
                print(f"Dropped params: {', '.join(dropped_keys)}")

            try:
                r = requests.post(
                    url=full_url,
                    json=filtered_body,
                    headers=headers,
                    stream=False,
                )
                r.raise_for_status()

                data = r.json()
                content = ""
                if (
                    isinstance(data, dict)
                    and "choices" in data
                    and isinstance(data["choices"], list)
                    and len(data["choices"]) > 0
                    and "message" in data["choices"][0]
                    and "content" in data["choices"][0]["message"]
                ):
                    content = data["choices"][0]["message"]["content"]
                else:
                    content = str(data)

                def chunk_text(text: str, chunk_size: int = 30) -> Generator[str, None, None]:
                    for i in range(0, len(text), chunk_size):
                        yield text[i : i + chunk_size]

                def fake_stream() -> Generator[str, None, None]:
                    for chunk in chunk_text(content):
                        yield chunk

                return fake_stream()

            except Exception as e:
                if "r" in locals() and r is not None:
                    return f"Error: {e} ({r.text})"
                else:
                    return f"Error: {e}"

        else:
            filtered_body = {k: v for k, v in body.items() if k in allowed_params_default}
            if len(body) != len(filtered_body):
                dropped_keys = set(body.keys()) - set(filtered_body.keys())
                print(f"Dropped params: {', '.join(dropped_keys)}")

            try:
                r = requests.post(
                    url=full_url,
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
                if "r" in locals() and r is not None:
                    return f"Error: {e} ({r.text})"
                else:
                    return f"Error: {e}"
