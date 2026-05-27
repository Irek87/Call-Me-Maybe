import json
from typing import List
import src.models as m
from src.parser import Parser
from llm_sdk.llm_sdk import Small_LLM_Model


def main() -> None:
    functions: List[m.Function] = []
    prompts: List[m.Prompt] = []

    try:
        f_defs, functions, prompts = Parser.init_parse()
        
        llm = Small_LLM_Model()
    except json.JSONDecodeError as e:
        print("An error during JSON decoding occured:\n", e, sep="")
    except Exception as e:
        print("Some error occured:\n", e, sep="")

    # print(llm.get_path_to_vocab_file())

    final_prompt = """<|im_start|>system
# Tools

You can choose only one function to assist with the user query.

You are provided with function definitions within <tools> and </tools> XML tags:
<tools> %s </tools>

Return a json object with function name and arguments following the schema within <tool_call> and </tool_call> XML tags:
<tool_call>
{"prompt": <user's prompt>, "name": <function-name>, "arguments": {<variable-name>: <value>}}
</tool_call><|im_end|>

<|im_start|>user
What is the sum of 2 and 3?
<|im_end|>

<|im_start|>assistant
""" % f_defs

    tensor = llm.encode(final_prompt).tolist()[0]
    tensor.extend(llm.encode("</think>").tolist()[0])
    # tensor.extend(llm.encode(' {"prompt": ').tolist()[0])

    answ_ = []
    # print("="*10)
    # print(tensor)
    for f_ in range(0, 100):
        logits = llm.get_logits_from_input_ids(tensor)
        max_logit = max(logits)
        i_max = 0
        # print(len(logits))
        for i in range(0, len(logits)):
            if logits[i] == max_logit:
                i_max = i
                break
        tensor.append(i_max)
        print(llm.decode([i_max]))
        answ_.append(i_max)
    print("="*30)
    # print(f_)
    print(llm.decode(answ_))


if __name__ == "__main__":
    main()
