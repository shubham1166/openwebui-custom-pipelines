from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        GEMINI_API_KEY: str
        GEMINI_ENDPOINT: str
        GEMINI_MODELS: str
        GEMINI_MODEL_NAMES: str

    def __init__(self):
        self.type = "manifold"
        self.name = "Gemni Vertex: "
        self.valves = self.Valves(
            **{
                "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here"),
                "GEMINI_ENDPOINT": os.getenv("GEMINI_ENDPOINT", "your-gemini-endpoint-here"),
                "GEMINI_MODELS": os.getenv("GEMINI_MODELS", "google/gemini-1.5-flash"),
                "GEMINI_MODEL_NAMES": os.getenv("GEMINI_MODEL_NAMES", "google/gemini-1.5-flash"),
            }
        )
        self.set_pipelines()

    def set_pipelines(self):
        models = self.valves.GEMINI_MODELS.split(";")
        model_names = self.valves.GEMINI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model, "name": name} for model, name in zip(models, model_names)
        ]
        print(f"gemini_manifold_pipeline - models: {self.pipelines}")

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
    ) -> Union[str, Generator[str, None, None], Iterator[str]]:
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        user_email = body.get("user", {}).get("email")

        body["model"] = model_id

        headers = {
            "api-key": self.valves.GEMINI_API_KEY,
            "Content-Type": "application/json",
        }

        # URL for Chat Completions in OpenAI
        url = (
            f"{self.valves.GEMINI_ENDPOINT}?source={user_email}"
        )

        # Ensure user is a string
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"].get("id", str(body["user"]))


        try:
            r = requests.post(
                url=url,
                json=body,
                headers=headers,
                stream=True,
            )
            r.raise_for_status()

            if body.get("stream"):
                # Real streaming
                return r.iter_lines()
            else:
                # Just return the JSON
                return r.json()

        except Exception as e:
            if "r" in locals() and r is not None:
                return f"Error: {e} ({r.text})"
            else:
                return f"Error: {e}"
