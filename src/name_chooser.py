import sys
from typing import List, Dict

from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import UserQuery
from src.prompts import get_func_name_prompt


class NameChooser():
    """Select a function name for a user query using token-level constraints.

    This helper drives the LLM token-by-token to produce a function name from a
    predefined set (plus a fallback `fn_NULL`).
    """

    @staticmethod
    def choose(
        llm: Small_LLM_Model,
        f_descriptions: str,
        u_q: UserQuery,
        valid_funcs_tok_ids: Dict[int, set[int]],
        f_names: List[str]
    ) -> str:
        """Choose a function name for the given user query.

        Args:
            llm: LLM wrapper used for tokenization and next-token logits.
            f_descriptions: Serialized function metadata (names/descriptions)
                included in the prompt context.
            u_q: User query object containing the prompt text.
            valid_funcs_tok_ids: Mapping from token position to the set of
            token IDs allowed at that position for function-name decoding.
            f_names: List of valid function names.

        Returns:
            The chosen function name if successful; otherwise an empty string.

        Note:
            This method prints progress to stdout.
        """

        func_name_tokens: List[int] = []
        func_name_prompt: str = get_func_name_prompt(f_descriptions,
                                                     u_q.prompt)
        prompt_tokens: List[int] = (llm.encode(func_name_prompt)
                                    .flatten()
                                    .tolist())
        prompt_tokens.extend(llm.encode("</think>").flatten().tolist())

        id_idx: int = 0
        func_name_str: str = ""

        while (1):
            logits = llm.get_logits_from_input_ids(
                prompt_tokens + func_name_tokens
            )

            if id_idx < len(valid_funcs_tok_ids):
                cur_step_ids = {
                    id: logits[id] for id
                    in valid_funcs_tok_ids[id_idx]
                }

            max_id = max(cur_step_ids, key=lambda k: cur_step_ids[k])
            func_name_tokens.append(max_id)
            func_name_str += llm.decode([max_id])
            print("name:", func_name_str, end='\r')

            if (func_name_str.startswith("fn_NULL")
                    or id_idx >= len(valid_funcs_tok_ids)):
                sys.stdout.write("\033[K")
                print("Could not find a suitable function\n")
                return ""
            if func_name_str in f_names:
                break
            id_idx += 1

        return func_name_str
