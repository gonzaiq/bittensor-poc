import asyncio
import random
from duckduckgo_search import AsyncDDGS, DDGS
import trafilatura
from threading import Semaphore
import concurrent.futures
import time


class SimultaneousRequestLimitError(Exception):
    pass


class WebSearchAPI:
    def __init__(
        self,
        error_rate: float = 0.2,
        empty_rate: float = 0.1,
        max_concurrent_requests: int = 5,
    ):
        self.error_rate = error_rate
        self.empty_rate = empty_rate
        self.async_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.sync_semaphore = Semaphore(max_concurrent_requests)
        self.request_count = 0

    async def async_search_and_scrape(
        self, query: str, n_results: int = 3
    ) -> list[str]:
        if self.async_semaphore.locked():
            raise SimultaneousRequestLimitError(
                "Maximum number of simultaneous requests reached. Please try again later."
            )
        async with self.async_semaphore:
            self.request_count += 1
            if random.random() < self.error_rate:
                raise Exception("Random API error occurred")
            if random.random() < self.empty_rate:
                return []
            try:
                return await self._original_async_search_and_scrape(query, n_results)
            except Exception as e:
                print(f"Error in original function: {str(e)}")
                return []

    async def _original_async_search_and_scrape(
        self, query: str, n_results: int = 3
    ) -> list[str]:
        async def fetch_content(url):
            try:
                html = trafilatura.fetch_url(url=url)
                content = trafilatura.extract(html)
                return {"url": url, "content": content}
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                return None

        with AsyncDDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=n_results + 2)]
        tasks = [fetch_content(result["href"]) for result in results[: n_results + 2]]
        contents = await asyncio.gather(*tasks)
        return [content["content"] for content in contents if content is not None][
            :n_results
        ]

    def search_and_scrape(self, query: str, n_results: int = 3) -> list[str]:
        if not self.sync_semaphore.acquire(blocking=False):
            raise SimultaneousRequestLimitError(
                "Maximum number of simultaneous requests reached. Please try again later."
            )
        try:
            self.request_count += 1
            random.seed()  # Ensure thread-safe random numbers
            if random.random() < self.error_rate:
                raise Exception("Random API error occurred")
            if random.random() < self.empty_rate:
                return []
            return self._original_sync_search_and_scrape(query, n_results)
        except Exception as e:
            print(f"Error in original function: {str(e)}")
            return []
        finally:
            self.sync_semaphore.release()

    def _original_sync_search_and_scrape(
        self, query: str, n_results: int = 3
    ) -> list[str]:
        def fetch_content(url):
            try:
                html = trafilatura.fetch_url(url=url)
                content = trafilatura.extract(html)
                return {"url": url, "content": content}
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                return None

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n_results + 2))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=n_results + 2
        ) as executor:
            futures = [
                executor.submit(fetch_content, result["href"])
                for result in results[: n_results + 2]
            ]
            contents = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        return [content["content"] for content in contents if content is not None][
            :n_results
        ]

    async def async_get_request_count(self) -> int:
        return self.request_count

    def get_request_count(self) -> int:
        return self.request_count

    async def async_reset_request_count(self) -> None:
        self.request_count = 0

    def reset_request_count(self) -> None:
        self.request_count = 0


web_search_api = WebSearchAPI()
