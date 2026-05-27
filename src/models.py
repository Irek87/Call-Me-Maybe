from pydantic import BaseModel, Field
from typing import Dict


class Function(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class Prompt(BaseModel):
    body: str = Field(min_length=1)
