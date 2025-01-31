import asyncio
from pydantic import BaseModel, model_validator
import re
from miner import Miner
from synapse import Synapse
import numpy as np
from loguru import logger
import pandas as pd
from chat_api import openai_api
from typing import Any
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import inspect

df = pd.read_csv("./dataset.csv")
challenges, references = [x for x in list(df["challenge"])], [
    x.strip() for x in list(df["reference"])
]

RUNTIME = 20


class Validator(BaseModel):
    miner: Any
    reward: float = 0
    step: int = 0
    tasks: list = []
    failed_responses: list = []
    successful_responses: list = []
    show_logs: bool = True
    executor: ProcessPoolExecutor | None = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def init_executor(self) -> "Validator":
        self.executor = ProcessPoolExecutor(max_workers=100)
        return self

    async def run_step(self):
        self.step += 1
        if self.show_logs:
            logger.info(f"Requesting miner on step {self.step}")
        reference, query = self._generate_task()
        synapse = Synapse(query=query)
        if not inspect.iscoroutinefunction(Miner().forward):
            task = self.executor.submit(self.miner.forward, synapse)
        else:
            task = asyncio.create_task(self.miner.forward(synapse=deepcopy(synapse)))
        self.tasks.append((task, reference, query))
        return query, reference

    def _generate_task(self) -> tuple[str, str]:
        i = np.random.randint(0, len(challenges))
        return references[i], challenges[i]

    async def score_responses(
        self, references: list[str], synapses: list[Synapse]
    ) -> list[float]:
        res_embs = await openai_api.async_get_embeddings([s.response for s in synapses])
        ref_embs = await openai_api.async_get_embeddings(references)
        if [s.response for s in synapses] and references and not (res_embs or ref_embs):
            raise Exception(
                "We couldn't score your responses due to connection issues. Please get in touch with your interviewers to resolve the problem!"
            )
        scores = []
        for res_emb, ref_emb in zip(res_embs, ref_embs):
            scores.append(
                (res_emb @ ref_emb)
                / (np.linalg.norm(ref_emb) * np.linalg.norm(res_emb))
            )
        for i, ref in enumerate(references):
            if len(ref) < 7:
                scores[i] = (
                    1
                    if re.sub("[., \n]", "", ref)
                    == re.sub("[., \n]", "", synapses[i].response)
                    else 0
                )
        return scores

    async def start(self):
        logger.info("Starting validator")
        end_time = asyncio.get_event_loop().time() + RUNTIME  # Run for 20 seconds
        interval = 0.2  # Run step every 0.5 seconds

        while asyncio.get_event_loop().time() < end_time:
            await asyncio.sleep(interval)
            if self.miner.busy:
                logger.info("Miner busy - skipping request")
                continue
            start_time = asyncio.get_event_loop().time()
            await self.run_step()
            elapsed = asyncio.get_event_loop().time() - start_time
            # sleep_time = max(0, interval - elapsed)

        logger.info("Time's up, processing responses...")
        completed_tasks = []
        completed_references = []
        completed_steps = []

        step = 0
        for task, reference, query in self.tasks:
            step += 1
            if task.done():
                if not inspect.iscoroutinefunction(Miner().forward):
                    synapse = task.result()
                else:
                    synapse = await asyncio.wait_for(
                        task, timeout=0.01
                    )  # Short timeout to handle hanging tasks
                try:
                    if not isinstance(synapse, Synapse):
                        raise Exception(
                            "Your 'forward' function must return a Synapse object"
                        )
                    if not isinstance(synapse.response, str):
                        raise Exception(
                            f"Your synapse.response variable should be string but is {type(synapse.response)}"
                        )
                    if synapse.query != query:
                        raise Exception("Task not equal to query...")
                    if len(synapse.response) == 0:
                        self.failed_responses.append(
                            {
                                "step": step,
                                "query": query,
                                "error": "Empty response",
                                "reason": "The miner returned an empty response (please set synapse.response to your response)",
                            }
                        )
                    else:
                        completed_tasks.append(synapse)
                        completed_steps.append(step)
                        completed_references.append(reference)
                except asyncio.TimeoutError:
                    self.failed_responses.append(
                        {
                            "step": step,
                            "query": query,
                            "error": "Timeout",
                            "reason": "The task timed out while waiting for a response",
                        }
                    )
                except Exception as e:
                    self.failed_responses.append(
                        {
                            "step": step,
                            "query": query,
                            "error": "Exception",
                            "reason": f"An exception occurred: {str(e)}",
                        }
                    )
            else:
                self.failed_responses.append(
                    {
                        "step": step,
                        "query": query,
                        "error": "Timeout",
                        "reason": f"The task was not completed within the {RUNTIME}-second timeframe",
                    }
                )

        if completed_tasks:
            scores = await self.score_responses(completed_references, completed_tasks)
            self.reward = sum(scores)

            for step, synapse, reference, score in zip(
                completed_steps,
                completed_tasks,
                completed_references,
                scores,
            ):
                self.successful_responses.append(
                    {
                        "step": step,
                        "query": synapse.query,
                        "reference": reference,
                        "response": synapse.response,
                        "score": score,
                    }
                )

        logger.info("Failed Responses:")
        for failed in self.failed_responses:
            logger.error(f"Step: {failed['step']}")
            logger.error(f"Query: {failed['query']}")
            logger.error(f"Error: {failed['error']}")
            logger.error(f"Reason: {failed['reason']}")
            logger.error("---")

        logger.info("Successful Responses:")
        for success in self.successful_responses:
            logger.info(f"Step: {success['step']}")
            logger.info(f"Query: {success['query']}")
            logger.info(f"Reference: {success['reference']}")
            logger.info(f"Response: {success['response']}")
            logger.info(f"Score: {success['score']}")
            logger.info("---")
        logger.info(
            f"Responded to {len(completed_tasks)} out of {len(self.tasks)} total tasks within the {RUNTIME} second timeframe"
        )

        logger.info(f"Miner achieved reward of: {self.reward}")
        logger.info(f"Total steps: {self.step}")


async def main():
    validator = Validator(miner=Miner())
    logger.info("Validator created")
    await validator.start()


if __name__ == "__main__":
    asyncio.run(main())
