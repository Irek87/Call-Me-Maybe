from typing import Dict, Any


def get_func_name_prompt(f_descriptions: str, user_query: str) -> str:
    """Build a chat-formatted prompt to select a function name.

    Args:
        f_descriptions: Serialized list of available functions containing at
            least names and descriptions.
        user_query: The raw user prompt to route to a function.

    Returns:
        A single string containing a chat-formatted prompt intended to make the
        model respond with only a suitable function name.
    """

    return ("<|im_start|>system\n" +
            "You are a function selector.\n" +
            f"Available functions:\n{f_descriptions}.\n" +
            "Reply with a suitable function name<|im_end|>" +
            f"<|im_start|>user {user_query}<|im_end|>" +
            "<|im_start|>assistant")


def get_param_prompt(func_name: str, f_def: Dict[str, Any],
                     user_query: str) -> str:
    """Build a chat-formatted prompt to generate function parameters as JSON.

    The prompt embeds the selected function definition and instructs the model
    to output a JSON object of the form `{"parameters": {...}}` consistent with
    the function's parameter schema.

    Args:
        func_name: Selected function name.
        f_def: Function definition dictionary containing at least `description`
            and `parameters` keys.
        user_query: The raw user prompt to extract parameter values from.

    Returns:
        A single string containing a chat-formatted prompt for parameter
        extraction.
    """

    return ("<|im_start|>system\n" +
            "Here is a function definition to use:\n{" +
            f"\"name\": \"{func_name}\", " +
            f"\"description\": \"{f_def['description']}\", " +
            f"\"parameters\": {f_def['parameters']}" +
            "}\nMake a valid json format with escaping backslashes, " +
            "taking into account types of paramaeters, " +
            "according to the schema: " +
            "{\"parameters\": {<variable-name>: <value>}}\n<|im_end|>" +
            f"<|im_start|>user {user_query}<|im_end|>" +
            "<|im_start|>assistant")
