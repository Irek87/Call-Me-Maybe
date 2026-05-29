from pydantic import BaseModel, Field
from typing import Dict, Any


# class Parameter(BaseModel):
#     name: str = Field(min_length=1)
#     value: Any = Field()


class Function(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class UserQuery(BaseModel):
    prompt: str = Field(min_length=1)


class OuputItem(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
