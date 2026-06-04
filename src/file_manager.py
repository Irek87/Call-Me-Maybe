import argparse
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path
from src.models import Function, UserQuery


class FileManager():
    """Handle CLI argument parsing and JSON I/O for the application.

    This class is responsible for:
      - Parsing command-line arguments that configure input/output paths.
      - Loading function definitions and user queries from JSON files.
      - Writing the final results to a JSON output file.

    Attributes:
        args: Parsed CLI arguments (functions_definition, input, output).
    """

    args = argparse.Namespace()

    @classmethod
    def fill_args(cls) -> None:
        """Parse CLI arguments and store them on the class.

        Populates `FileManager.args` with:
          - functions_definition: path to function definitions JSON
          - input: path to user prompts JSON
          - output: path to output results JSON
        """

        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition",
                            default="data/input/functions_definition.json",
                            type=str,
                            help="A path of functions' definitions")
        parser.add_argument("--input",
                            default="data/input/function_calling_tests.json",
                            type=str,
                            help="A path of promts")
        parser.add_argument("--output",
                            default="data/output/function_calls.json",
                            type=str,
                            help="A path where to save results")
        cls.args = parser.parse_args()

    @classmethod
    def init_parse(cls) -> Tuple[List[Function], List[UserQuery]]:
        """Load and validate inputs from JSON files.

        Reads function definitions and user queries
        from the paths in `cls.args`.
        Performs basic validation (non-empty lists, no empty items, unique
        function names) and converts raw dictionaries to strongly-typed models.

        Returns:
            A tuple `(functions, user_queries)`.

        Raises:
            FileNotFoundError: If an input file path does not exist.
            json.JSONDecodeError: If an input file contains invalid JSON.
            KeyError: If expected keys are missing in the JSON payload.
            Exception: If semantic validation fails
                (e.g., duplicates, empties).
        """

        functions: List[Function] = []
        user_queries: List[UserQuery] = []

        cls.fill_args()

        with open(cls.args.functions_definition) as f_defs_raw:
            f_defs = json.loads(f_defs_raw.read())
            if not f_defs:
                raise Exception("There is no function definition")
            if not all(f_defs):
                raise Exception("There is an empty function definition")
            f_names = [f_def["name"] for f_def in f_defs]
            if len(f_names) != len(set(f_names)):
                raise Exception("There are duplicate functions")
            for f_def in f_defs:
                functions.append(
                    Function(name=f_def["name"],
                             description=f_def["description"],
                             parameters=f_def["parameters"],
                             returns=f_def["returns"])
                )

        with open(cls.args.input) as queries_raw:
            queries = json.loads(queries_raw.read())
            if not queries:
                raise Exception("There are no prompts")
            if not all(queries):
                raise Exception("There is an empty prompt")
            for query in queries:
                user_queries.append(UserQuery(prompt=query["prompt"]))

        return (functions, user_queries)

    @classmethod
    def write_output(cls, results: List[Dict[str, Any]]) -> None:
        """Write results to the configured output path as pretty-printed JSON.

        Ensures the output directory exists and serializes `results` to a JSON
        file using `indent=2`.

        Args:
            results: List of JSON-serializable result items to write.

        Raises:
            OSError: If the output path cannot be created or written.
            TypeError: If `results` contains non-JSON-serializable objects.
        """

        Path(cls.args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(cls.args.output, "w") as f:
            json.dump(results, f, indent=2)
