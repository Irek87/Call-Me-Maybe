import json
from typing import List
import src.models as m
from src.parser import Parser
from llm_sdk.llm_sdk import Small_LLM_Model


def get_func_name_prompt(f_defs: str, user_query: str) -> str:
    return """<|im_start|>system
    You can choose only one function to assist with the user query if there is an option.
    You are provided with function definitions within <tools> and </tools> XML tags:
    <tools> %s </tools>
    Please show your choice in your response, returning only a function name. If none of the functions are suitable, return just one word "NULL".
    <|im_end|>
    <|im_start|>user
    %s
    <|im_end|>
    <|im_start|>assistant
    """ % (f_defs, user_query)


def get_func_names(f_defs: str, user_queries: List[m.UserQuery], llm: Small_LLM_Model) -> List[str]:
    anim_idx = 0
    func_names = []
    animation = [
        "|=       |",
        "|===     |",
        "|====    |",
        "|=====   |",
        "|======  |",
        "|======= |",
        "|========|",
        "| =======|",
        "|  ======|",
        "|   =====|",
        "|    ====|",
        "|     ===|",
        "|      ==|",
        "|       =|",
        "|        |"
    ]

    for u_q in user_queries:
        answ = []
        func_name_prompt = get_func_name_prompt(f_defs, u_q)
        tensor = llm.encode(func_name_prompt).flatten().tolist()
        tensor.extend(llm.encode("</think>\n```python\n").flatten().tolist())

        for f_ in range(0, 100):
            logits = llm.get_logits_from_input_ids(tensor)
            max_logit = max(logits)
            i_max = 0
            for i in range(0, len(logits)):
                if logits[i] == max_logit:
                    i_max = i
                    break
            if i_max > 151640 or i_max == 198:
                func_names.append(llm.decode(answ))
                break
            tensor.append(i_max)
            
            # print(llm.decode([i_max]))
            print(animation[anim_idx % len(animation)], end="\r")
            anim_idx += 1
            
            answ.append(i_max)

    return func_names


def main() -> None:
    functions: List[m.Function] = []
    user_queries: List[m.UserQuery] = []

    try:
        f_defs, functions, user_queries = Parser.init_parse()
        
        llm = Small_LLM_Model()
    except json.JSONDecodeError as e:
        print("An error during JSON decoding occured:\n", e, sep="")
    except Exception as e:
        print("Some error occured:\n", e, sep="")

    # with open(llm.get_path_to_vocab_file()) as v:
    #     print(v.read()[-55:])

    func_names = get_func_names(f_defs, user_queries, llm)
    print("="*30)
    print(func_names)


if __name__ == "__main__":
    main()





# final_prompt = """<|im_start|>system
# # Tools

# You can choose only one function to assist with the user query.

# You are provided with function definitions within <tools> and </tools> XML tags:
# <tools> %s </tools>

# Return a json object with function name and arguments following the schema within <tool_call> and </tool_call> XML tags:
# <tool_call>
# {
#     "prompt": <user's prompt>,
#     "name": <function-name>,
#     "parameters": {<variable-name>: <value>}
# }
# </tool_call><|im_end|>

# <|im_start|>user
# Greet shrek
# <|im_end|>

# <|im_start|>assistant
# """ % f_defs