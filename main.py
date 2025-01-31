from miner import Miner
from validator import Validator
import asyncio
from loguru import logger
from miner import Miner, SHOW_VALIDATOR_LOGS


async def main():
    validator = Validator(miner=Miner(), show_logs=SHOW_VALIDATOR_LOGS)
    logger.info("Validator created")
    await validator.start()


if __name__ == "__main__":
    asyncio.run(main())
