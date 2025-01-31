from pydantic import BaseModel


class Synapse(BaseModel):
    query: str
    response: str = ""
