import argparse
import json
from typing import List, Tuple
import src.models as m


class Parser():
    @staticmethod
    def init_parse() -> Tuple[str, List[m.Function], List[m.Prompt]]:
        functions: List[m.Function] = []
        user_queries: List[m.UserQuery] = []
        f_defs: str = ""
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=str, required=True,
                            help="A path of functions' definitions")
        parser.add_argument("--input", type=str, required=True,
                            help="A path of promts")
        parser.add_argument("--output", type=str, required=True,
                            help="A path where to save results")

        args = parser.parse_args()

        with open(args.functions_definition) as f_def_raw:
            f_def = json.loads(f_def_raw.read())
            for f in f_def:
                f_defs += str(f) + " "
                functions.append(
                    m.Function(name=f["name"],
                               description=f["description"],
                               parameters=f["parameters"],
                               returns=f["returns"]))
        with open(args.input) as queries_raw:
            queries = json.loads(queries_raw.read())
            for query in queries:
                user_queries.append(m.UserQuery(prompt=query["prompt"]))

        return (f_defs, functions, user_queries)
