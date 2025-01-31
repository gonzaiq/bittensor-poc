import asyncio
import random
import time
from typing import Optional
import openai
import dotenv
from loguru import logger
import os
from openai.types import CreateEmbeddingResponse
import numpy as np
from threading import Semaphore

if not dotenv.load_dotenv():
    logger.error("Couldn't load env file")

async_client = openai.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))
sync_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))


class SimultaneousRequestLimitError(Exception):
    pass


class OpenAIAPIWrapper:
    def __init__(
        self,
        error_rate: float = 0.1,
        empty_rate: float = 0.05,
        max_simultaneous_requests: int = 5,
        retries: int = 3,
    ):
        self.error_rate = error_rate
        self.empty_rate = empty_rate
        self.async_semaphore = asyncio.Semaphore(max_simultaneous_requests)
        self.sync_semaphore = Semaphore(max_simultaneous_requests)
        self.request_count = 0
        self.retries = retries

    async def async_get_openai_response(self, prompt: str) -> str:
        if self.async_semaphore.locked():
            raise SimultaneousRequestLimitError(
                "Maximum number of simultaneous requests reached. Please try again later."
            )
        async with self.async_semaphore:
            self.request_count += 1
            if random.random() < self.error_rate:
                raise Exception("Random API error occurred")
            if random.random() < self.empty_rate:
                return ""
            try:
                return await self._original_get_openai_response_async(prompt)
            except Exception as e:
                logger.error(f"Error in original function: {str(e)}")
                return "Failed to get a response after multiple retries."

    async def _original_get_openai_response_async(self, prompt: str) -> str:
        await asyncio.sleep(1)
        for _ in range(self.retries):
            try:
                response = await async_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
        return "Failed to get a response after multiple retries - this is an issue with the actual openai api and not part of the test. You can ignore this and pretend this is an actual response from openai."

    def get_openai_response(self, prompt: str) -> str:
        if not self.sync_semaphore.acquire(blocking=False):
            raise SimultaneousRequestLimitError(
                "Maximum number of simultaneous requests reached. Please try again later."
            )
        if random.random() < self.error_rate:
            raise Exception("Random API error occurred")
        self.request_count += 1
        if random.random() < self.empty_rate:
            return ""
        try:

            return self._original_get_openai_response_sync(prompt)
        except Exception as e:
            logger.error(f"Error in original function: {str(e)}")
            return "Failed to get a response after multiple retries."
        finally:
            self.sync_semaphore.release()

    def _original_get_openai_response_sync(self, prompt: str) -> str:
        time.sleep(1)
        for _ in range(self.retries):
            try:
                response = sync_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
        return "Failed to get a response after multiple retries."

    async def async_get_embeddings(self, texts: list[str]):
        for _ in range(3):
            try:
                result: CreateEmbeddingResponse = await async_client.embeddings.create(
                    input=texts, model="text-embedding-3-small"
                )
                return [np.array(x.embedding) for x in result.data]
            except Exception as ex:
                logger.error(
                    f"Exception whilst getting embeddings (likely due to connection issues, get in touch with your interviewer): {ex}"
                )

    def get_embeddings(self, texts: list[str]):
        for _ in range(3):
            try:
                result: CreateEmbeddingResponse = sync_client.embeddings.create(
                    input=texts, model="text-embedding-3-small"
                )
                return [np.array(x.embedding) for x in result.data]
            except Exception as ex:
                logger.debug(f"Exception whilst getting embeddings: {ex}")

    async def async_get_request_count(self) -> int:
        return self.request_count

    def get_request_count(self) -> int:
        return self.request_count

    async def async_reset_request_count(self) -> None:
        self.request_count = 0

    def reset_request_count(self) -> None:
        self.request_count = 0


openai_api = OpenAIAPIWrapper(max_simultaneous_requests=5)
