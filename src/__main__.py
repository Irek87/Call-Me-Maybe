import json
from typing import List, Dict, Any

from llm_sdk.llm_sdk import Small_LLM_Model
from src.utils import timer
from src.file_manager import FileManager
from src.name_chooser import NameChooser
from src.params_extractor import ParamsExtractor
from src.models import Function, UserQuery


@timer
def get_results(user_queries: List[UserQuery],
                functions: List[Function],
                llm: Small_LLM_Model) -> List[Dict[str, Any]]:
    """Generate function-call results for a list of user queries.

    For each query, this function:
      1) Selects a function name using the LLM (via `NameChooser`).
      2) Extracts parameters for the selected function (via `ParamsExtractor`).
      3) Appends a result item with keys: `prompt`, `name`, `parameters`.

    Notes:
        This function currently mutates the `functions` list by appending a
        fallback `fn_NULL` function.

    Args:
        user_queries: User prompts to process.
        functions: Available tool/function definitions.
        llm: LLM wrapper used for tokenization and next-token logits.

    Returns:
        A list of JSON-serializable dictionaries,
        one per successfully processed query, with the schema:
        `{"prompt": str, "name": str, "parameters": dict}`.
    """

    results: List[Dict[str, Any]] = []

    functions.append(
        Function(
            name="fn_NULL",
            description="If no other function is suitable for user's query",
            parameters={},
            returns={}
        )
    )

    f_names: List[str] = [f.name for f in functions]
    f_descriptions: str = str([
        {"name": f.name, "description": f.description} for f in functions
    ])
    valid_funcs_tok_ids: Dict[int, set[int]] = {}
    for f_name in f_names:
        func_tok_ids = llm.encode(f_name).flatten().tolist()
        for i in range(len(func_tok_ids)):
            valid_funcs_tok_ids[i] = valid_funcs_tok_ids.setdefault(i, set())
            valid_funcs_tok_ids[i].add(func_tok_ids[i])

    print()
    for u_q in user_queries:
        print("prompt:", u_q.prompt)
        func_name = NameChooser.choose(llm,
                                       f_descriptions,
                                       u_q,
                                       valid_funcs_tok_ids,
                                       f_names)

        if func_name:
            print()
            f_def = [
                {"description": f.description, "parameters": f.parameters}
                for f in functions if f.name == func_name
            ]
            if not f_def:
                print(
                    "Could not fetch parameters " +
                    "from the function definition"
                )
                continue
            params = ParamsExtractor.get_params(func_name,
                                                f_def[0],
                                                u_q.prompt,
                                                llm)
            if params:
                print()
                results.append(
                    {
                        "prompt": u_q.prompt,
                        "name": func_name,
                        "parameters": params["parameters"]
                    }
                )
            print()

    return results


def main() -> None:
    """Run the application entry point.

    Loads function definitions and user queries, initializes the LLM,
    generates function-call results, and writes the output JSON file.

    Raises:
        json.JSONDecodeError: If an input JSON file cannot be decoded.
        Exception: For any other unexpected runtime error.
    """

    try:
        functions, user_queries = FileManager.init_parse()
        llm = Small_LLM_Model()
        results = get_results(user_queries, functions, llm)
        FileManager.write_output(results)
    except json.JSONDecodeError as e:
        print("An error during JSON decoding occured:\n", e, sep="")
    except Exception as e:
        print("Some error occured:\n", e, sep="")


if __name__ == "__main__":
    main()
