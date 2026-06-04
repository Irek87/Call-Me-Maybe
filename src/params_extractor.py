import json
from typing import Dict, Any

from llm_sdk.llm_sdk import Small_LLM_Model
from src.prompts import get_param_prompt


class ParamsExtractor():
    """Extract function-call parameters for a selected function using the LLM.

    The current implementation performs greedy decoding to generate a JSON
    object of the form `{"parameters": {...}}`, attempting to parse the JSON
    after each generated token until it becomes valid.
    """

    @staticmethod
    def get_params(func_name: str,
                   f_def: Dict[str, Any],
                   user_query: str,
                   llm: Small_LLM_Model
                   ) -> Dict[str, Any]:
        """Generate and parse a parameters JSON object for the given function.

        Args:
            func_name: Selected function name for which to generate parameters.
            f_def: Function definition containing parameter schema metadata.
            user_query: Original user prompt text.
            llm: LLM wrapper used for tokenization and next-token logits.

        Returns:
            A parsed JSON dictionary expected to contain a top-level
            `"parameters"` key. Returns an empty dict on failure (e.g., token
            limit reached before producing valid JSON).

        Raises:
            KeyError: If `f_def` does not contain
                the expected `"parameters"` key.
            StopIteration: If the function schema has no parameters and
                `next(iter(f_def["parameters"]))` is invoked.

        Note:
            This method prints progress to stdout and uses
            a token limit to prevent unbounded generation.
        """

        params: Dict[str, Any] = {}
        max_tokens = 777
        tokens_counter = 0

        param_tokens = llm.encode('{"parameters": {').flatten().tolist()
        first_param_name = next(iter(f_def["parameters"]))
        param_tokens.extend(
            llm.encode(f'"{first_param_name}":').flatten().tolist()
        )

        param_prompt = get_param_prompt(func_name, f_def, user_query)
        param_prompt_tokens = llm.encode(param_prompt).flatten().tolist()
        param_prompt_tokens.extend(
            llm.encode("</think>\n```json\n").flatten().tolist()
        )

        while (1):
            logits = llm.get_logits_from_input_ids(param_prompt_tokens
                                                   + param_tokens)
            i_max = logits.index(max(logits))
            param_tokens.append(i_max)
            params_str = (llm.decode(param_tokens).replace('\n', ''))
            print(params_str, end='\r')
            tokens_counter += 1
            try:
                params = json.loads(params_str)
                break
            except json.JSONDecodeError:
                if tokens_counter >= max_tokens:
                    print("\nFailed to retrieve parameters " +
                          "within the token limit.")
                    return {}
                continue
        return params
