import argparse
import json
from typing import List, Tuple, Dict, Any
import src.models as m


class Parser():
    @staticmethod
    def init_parse() -> Tuple[str, List[m.Function], List[m.Prompt]]:
        functions: List[m.Function] = []
        prompts: List[m.Prompt] = []
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
                f_defs += str(f) #+ " "
                functions.append(
                    m.Function(name=f["name"],
                               description=f["description"],
                               parameters=f["parameters"],
                               returns=f["returns"]))
        with open(args.input) as inp_raw:
            inp = json.loads(inp_raw.read())
            for i in inp:
                prompts.append(m.Prompt(body=i["prompt"]))

        return (f_defs, functions, prompts)
