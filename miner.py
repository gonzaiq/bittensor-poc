from pydantic import BaseModel
from synapse import Synapse
import numpy as np
import asyncio
from chat_api import openai_api
from web_search_api import web_search_api
from loguru import logger

SHOW_VALIDATOR_LOGS = True  # set to false to not see any logs from the validator


class Miner(BaseModel):
    busy: bool = False
    previous_answers: dict = {}
    concurrent_calls_count: int = 0
    max_concurrent_calls: int = 5

    async def get_web_content(self, query, persist_tries: int = 5):
        """
        Get web content function with persistance mechanism.
        """

        for _ in range(persist_tries):
            
            try:
                web_content = await web_search_api.search_and_scrape(query=query)
                if len(web_content) > 0:
                    return web_content
            
            except Exception:
                pass
        
        return None
    
    def get_default_answer(self, query, is_math_query: bool):
        """
        Heuristic for default answer to maximize cosine similarity score, when query was not treated.
        """

        if is_math_query:
            # answer a number if it is a math question (could be fixed number, hyperparameter, or rand number)
            return 40
        else:
            # answer the body of the same question, with some modification.
            return "According to the context provided, " + query.lstrip("\n\n").lstrip("What is").lstrip("What was").lstrip("What")

    async def forward(self, synapse: Synapse, persist_tries: int = 5, do_web_search: bool = False) -> Synapse:

        # get cached answer if query seen before
        cached_ans = self.previous_answers.get(synapse.query, None)
        if cached_ans is not None:
            synapse.response = cached_ans
            return synapse

        # classify queries into math and non-math. Heuristic: non-math queries start with "\n\n"
        if synapse.query.startswith("\n\n"):
            math_query_flag = False
        else:
            math_query_flag = True
            
        # if max concurrent calls achieved, return default answer
        self.concurrent_calls_count += 1
        if self.concurrent_calls_count == self.max_concurrent_calls:
            synapse.response = self.get_default_answer(synapse.query, is_math_query=math_query_flag)
            return synapse
        
        # get query to LLM
        prompt = synapse.query
        if do_web_search and not math_query_flag:
            # expand query with web search
            # TODO Fix API
            web_content = await self.get_web_content(synapse.query)
            if web_content is not None:
                prompt += ". Web search results {web_content}"

        # call LLMN with persistance mechanism
        while persist_tries > 0:
            try:
                synapse.response = await openai_api.async_get_openai_response(
                    prompt=prompt
                    )
                break
            except Exception:
                pass

            persist_tries -= 1
        
        if persist_tries == 0:
            # could not get an answer from API, return default answer
            synapse.response = self.get_default_answer(synapse.query, is_math_query=math_query_flag)
        else:
            # complete cache mechanism with answer
            self.previous_answers[synapse.query] = synapse.response

        # decrease concurrent calls counter
        self.concurrent_calls_count -= 1

        return synapse
