import json
from typing import List, Dict, Any
import src.models as m
from src.parser import Parser
from llm_sdk.llm_sdk import Small_LLM_Model


def get_func_name_prompt(f_defs: str, user_query: str) -> str:
    # return """<|im_start|>system
    # You can only choose one function to assist with the user query
    # if there is such an option.
    # You are provided with function definitions within <tools> and </tools>
    # XML tags:
    # <tools> %s </tools>
    # You have to show your choice, returning only a function name.
    # If you are not sure that a function is suitable,
    # answer with just one word "fn_NULL".
    # <|im_end|>
    # <|im_start|>user
    # %s
    # <|im_end|>
    # <|im_start|>assistant
    # """ % (f_defs, user_query)
    return """<|im_start|>system
    Here are function definitions [ %s ].
    You have to choose one function name.
    If no function is suitable, answer just "fn_NULL".
    <|im_end|>
    <|im_start|>user
    %s<|im_end|>
    <|im_start|>assistant
    """ % (f_defs, user_query)


def get_result(f_defs: str, user_queries: List[m.UserQuery], llm: Small_LLM_Model, functions: List[m.Function]) -> List:
    
    valid_funcs_tok_ids: Dict[int, set[int]] = {}
    for f in functions:
        func_tok_ids = llm.encode(f.name).flatten().tolist()
        for i in range(len(func_tok_ids)):
            valid_funcs_tok_ids[i] = valid_funcs_tok_ids.setdefault(i, set())
            valid_funcs_tok_ids[i].add(func_tok_ids[i])

    result = []
    anim_idx = 0
    func_name = []
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
        tensor_list = llm.encode(func_name_prompt).flatten().tolist()
        # tensor_list.extend(llm.encode("</think>\n```python\n").flatten().tolist())
        tensor_list.extend(llm.encode("</think>").flatten().tolist())

        id_idx = 0

        while (1):
            logits = llm.get_logits_from_input_ids(tensor_list)
            
            cur_ids = {id:logits[id] for id in valid_funcs_tok_ids[id_idx]}
            # print(cur_ids)
            if id_idx < len(valid_funcs_tok_ids) - 1:
                id_idx += 1

            i_max = logits.index(max(logits))
            if i_max > 151640 or i_max == 198:
                # func_name = llm.decode(answ)
                # if func_name in f_defs:
                #     answ = []
                #     final_prompt = """<|im_start|>system
                # You only help me to make a json object.
                # Function definitions are [%s].
                # The 'prompt' to use in response is [%s].
                # Return a json object with 'prompt', the name of a function given by user and arguments, according to the schema within <tool_call> and </tool_call> XML tags:
                # <tool_call>{"prompt": <prompt>, "name": <function-name>, "parameters": {<variable-name>: <value>}}</tool_call>
                # <|im_end|>
                # <|im_start|>user
                # %s
                # <|im_end|>
                # <|im_start|>assistant
                # """ % (f_defs, u_q, func_name)
                    
                #     tensor_list = llm.encode(final_prompt).flatten().tolist()
                #     tensor_list.extend(llm.encode("</think>\n```json").flatten().tolist())

                #     for f_ in range(0, 100):
                #         logits = llm.get_logits_from_input_ids(tensor_list)
                #         i_max = logits.index(max(logits))
                #         if i_max > 151640 or i_max == 73594:
                #             result.append(llm.decode(answ))
                #             break
                #         tensor_list.append(i_max)
                #         answ.append(i_max)
                #         print("-" + llm.decode([i_max]) + f" : {i_max}")
                #         # print(animation[anim_idx % len(animation)], end="\r")
                #         anim_idx += 1
                result.append(llm.decode(answ))
                break
            
            # tensor_list.append(i_max)
            cur_max_id = max(cur_ids, key=cur_ids.get)
            tensor_list.append(cur_max_id)
            
            print(llm.decode([i_max]) + f" : {i_max}")
            # print(animation[anim_idx % len(animation)], end="\r")
            # anim_idx += 1
            # answ.append(i_max)
            answ.append(cur_max_id)

    return result


def get_inv_vocab(llm: Small_LLM_Model) -> Dict[str, Any]:
        with open(llm.get_path_to_vocab_file()) as vocab:
            vocab_dict = json.loads(vocab.read())
            return {v: k for k, v in vocab_dict.items()}


def main() -> None:
    functions: List[m.Function] = []
    user_queries: List[m.UserQuery] = []

    try:
        f_defs, functions, user_queries = Parser.init_parse()
        llm = Small_LLM_Model()
        # inverted_vocab = get_inv_vocab(llm)
    except json.JSONDecodeError as e:
        print("An error during JSON decoding occured:\n", e, sep="")
    except Exception as e:
        print("Some error occured:\n", e, sep="")



    import time
    start = time.time()

    result = get_result(f_defs, user_queries, llm, functions)
    elapsed_time = time.time() - start
    print("="*42)
    print(f"Elapsed time is {int(elapsed_time / 60)} min {int(elapsed_time % 60)} sec")
    print(result)


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