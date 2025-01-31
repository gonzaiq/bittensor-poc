
# bittensor-poc

Simulation of a Bittensor subnet, where validators send tasks two miner, miners accomplish them, and thei result is evaluated by validators. A final reward is then calculated (40-50 with this miner version).

Two types of tasks are considered: mathematical reasoning and question answering. Open AI's GPT is leveraged by the miner.

Fore more information on Bittensor, visit https://docs.bittensor.com/learn/introduction

## Repository Structure

<pre>
dataset.csv # tasks and groun-truth resuts
chat_api.py # wrapper for Open AI's api
web_search_api.py
synapse.py  # Script for training a contrastive model
validator.py 
miner.py 
main.py
demo.ipyng
.gitignore
.gitattributes  # Git attributes file for handling line endings and other settings
README.md  # This file
</pre>


## Setup and run

Install dependencies:

```bash
poetry install
```

Run main code. Validators send tasks to miners, that accomplish them and get rewarded:

```bash
poetry run python main.py
```