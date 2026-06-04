from pydantic import BaseModel, Field
from typing import Dict


class Function(BaseModel):
    """A callable tool/function definition parsed from the input schema.

    Attributes:
        name: Unique function identifier used in tool selection.
        description: Human-readable description of what the function does.
        parameters: Parameter schema mapping parameter name to a property map
            (e.g., type/description). The current structure is
            `Dict[param_name, Dict[str, str]]`.
        returns: Return schema description (structure depends on input format).
    """

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


class UserQuery(BaseModel):
    """A single user prompt to be routed to one of the available functions.

    Attributes:
        prompt: The natural-language user request to process.
    """

    prompt: str = Field(min_length=1)
