import asyncio
import json
from typing import Any, Dict, Optional

import httpx
import requests
from loguru import logger

from llama_index.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field
try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:
    from pydantic import Field, PrivateAttr


AUTH_URL = "the url for requesting autorization token."
ACCESS_KEY = "the access key for requesting autorization token."
OPEN_PLATFORM_API = "The API for using Your Open Platform given by supplier."


def time_wrapper(func):
    def wrapper(*args, **kwargs):
        import time

        # import inspect

        # print([t[3] for t in inspect.stack()])
        # print(args)
        # print(kwargs)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"LLM Time elapsed: {(end - start):.3f}")
        return result

    return wrapper


class AsyncClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=18)
        )
        # self._client = aiohttp.ClientSession()

    async def __aenter__(self):
        return self

    async def __aexit__(self):
        await self.close()

    async def close(self):
        await self._client.aclose()

    async def post(self, *args, **kwargs):
        return await self._client.post(*args, **kwargs)


class CusomHttpLLM(CustomLLM):
    model: str = Field(description="The model to use. Can be 'gpt4' or 'gpt-35-turbo'.")
    extend_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )
    environment: str = Field(
        description="The environment for the LLM API specified by supplier."
    )
    token: str = Field(description="The token for the LLM API specified by supplier.")
    OPEN_PLATFORM_API: str = Field(
        description="The API for the LLM specified by supplier."
    )
    _appKey: str = PrivateAttr()
    _appSecret: str = PrivateAttr()
    _access_key: str = PrivateAttr()
    _client: Optional[httpx.AsyncClient] = PrivateAttr()
    is_chat_model: bool = Field(description="Whether to use chat completion.")

    def __init__(
        self,
        appKey: str,
        appSecret: str,
        model: str = ("gpt4", "gpt-35-turbo"),
        environment: str = ("fat", "dev", "prod"),
        temperature: float = 0,
        topP: float = 1,
        stream: bool = False,
        presencePenalty: float = 0,
        frequencyPenalty: float = 0,
        maxTokens: int = 4096,
        num_output: int = 1,
        chat_mode: bool = True,
        **kwargs: Any,
    ):
        if model not in ("gpt4", "gpt-35-turbo"):
            raise ValueError("Invalid model! model can only in [gpt-35-turbo, gpt4]")
        if environment not in ("fat", "dev", "prod"):
            raise ValueError(
                "Invalid environment! environment can only in [fat, dev, prod]"
            )
        self._appKey = appKey
        self._appSecret = appSecret
        self.set_client()
        self._access_key = ACCESS_KEY
        token = self._get_token(environment)
        if token is None:
            raise ValueError("Failed to get token")

        extend_params = {
            "maxTokens": maxTokens,
            "temperature": temperature,
            "topP": topP,
            "n": num_output,
            "stream": stream,
            "presencePenalty": presencePenalty,
            "frequencyPenalty": frequencyPenalty,
        }
        super().__init__(
            model=model,
            token=token,
            environment=environment,
            extend_params=extend_params,
            is_chat_model=chat_mode,
            OPEN_PLATFORM_API=OPEN_PLATFORM_API,
            **kwargs,
        )

    def _get_token(self, environment: str):
        url = AUTH_URL
        header = {
            "x-gw-accesskey": self._access_key,
            "Content-Type": "application/json",
        }
        parameters = {"appKey": self._appKey, "appSecret": self._appSecret}
        try:
            response = requests.post(url, data=json.dumps(parameters), headers=header)
            return response.json()["data"] if response.status_code == 200 else None
        except Exception as e:
            print(e)
            return None

    def _call_headers(self):
        return {"X-GW-Authorization": self.token, "Content-Type": "application/json"}

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=2048,
            num_output=1,
            model_name="enn",
        )

    def _determine_url_api(self, prompt: str):
        if self.is_chat_model:
            url_api = f"{self.OPEN_PLATFORM_API}/chat/"
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            parameters = {
                "source": "Microsoft",
                "version": self.model,
                "messages": messages,
                "extendParams": self.extend_params,
            }
        else:
            url_api = f"{self.OPEN_PLATFORM_API}/completion"
            parameters = {
                "source": "Microsoft",
                "version": self.model,
                "prompt": prompt,
                "extendParams": self.extend_params,
            }
        return url_api, parameters

    # @time_wrapper
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        url_api, parameters = self._determine_url_api(prompt)
        parameters["extendParams"].update(kwargs)
        headers = self._call_headers()
        while True:
            text = ""
            response = {}
            try:
                response = requests.post(
                    url_api, data=json.dumps(parameters), headers=headers
                )
                if response.status_code != 200:
                    continue
                response = response.json()
                if response.get("code") == 704:
                    self.token = self._get_token(self.environment)
                    headers = self._call_headers()
                    continue
                try:
                    if self.is_chat_model:
                        text = response["data"]["choices"][0]["message"]["content"]
                    else:
                        text = response["data"]["choices"][0]["text"]
                    if len(text.strip()) == 0:
                        text = "####"
                    break
                except Exception as e:
                    print(f"text not in response {response}: \n", e)
                    continue
            except Exception as e:
                print("Get response error or response is not json: \n", e)
                continue
        return CompletionResponse(text=text, raw=response)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        if self.extend_params["stream"] is False:
            raise ValueError("Should specify stream=True for stream completion")

        def gen() -> CompletionResponseGen:
            text = ""
            url_api, parameters = self._determine_url_api(prompt)
            parameters["extendParams"].update(kwargs)
            headers = self._call_headers()
            try:
                responses = requests.post(
                    url_api, data=json.dumps(parameters), headers=headers
                )
                for response in responses.json()["data"]:
                    delta = response["choices"][0]["text"]
                    text += delta
                    yield CompletionResponse(
                        delta=delta,
                        text=text,
                        raw=response,
                    )
            except Exception as e:
                raise e

        return gen()

    def set_client(self, client: Optional[AsyncClient] = None):
        self._client = client or AsyncClient()

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return await self._acomplete(prompt, **kwargs)

    async def __async_post(self, url_api, parameters, headers):
        """
        Async http request with httpx
        """
        resp = await self._client.post(
            url_api, data=json.dumps(parameters), headers=headers
        )
        return None if resp.status_code != 200 else resp

    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        The aysnc http request for completion
        """
        url_api, parameters = self._determine_url_api(prompt)
        parameters["extendParams"].update(kwargs)
        headers = self._call_headers()
        while True:
            text = ""
            response = {}
            try:
                response = await self.__async_post(url_api, parameters, headers)
                if response is None:
                    continue
                response = response.json()

                if response.get("code") == 704:
                    self.token = self._get_token(self.environment)
                    headers = self._call_headers()
                    continue
                try:
                    if self.is_chat_model:
                        text = response["data"]["choices"][0]["message"]["content"]
                    else:
                        text = response["data"]["choices"][0]["text"]
                    if len(text.strip()) == 0:
                        text = "####"
                    break
                except Exception as e:
                    logger.debug(f"text not in response {response}: \n Exception: {e}")
                    continue
            except Exception:
                # logger.debug(
                #     f"Get response error or response is not json: \n Exception {e}"
                # )
                continue
        return CompletionResponse(text=text, raw=response)


if __name__ == "__main__":
    """
    Some test code for test the async request of CusomHttpLLM
    """

    import os

    def run_async_tasks(tasks):
        async def _gather():
            return await asyncio.gather(*tasks)

        return asyncio.run(_gather())

    app_key = os.environ.get("AppKey")
    app_secret = os.environ.get("AppSecret")
    llm = CusomHttpLLM(
        appKey=app_key,
        appSecret=app_secret,
        model="gpt4",
        environment="prod",
        temperature=1,
        topP=1,
    )
    res = run_async_tasks(
        [
            llm.acomplete("Please introduce yourself."),
            llm.acomplete("Do you know somthing about Sam Altman?"),
        ]
    )
    for r in res:
        print(r.json())
