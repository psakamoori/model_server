import numpy as np

def execute(inputs: dict) -> dict:
    print("Running execute method of the custom node")
    print(f"Received argument: {inputs}")
    input_arr = inputs["input"]
    input_arr += 1
    output_dict = {"output": input_arr}
    print(f"Returning {output_dict}")
    return output_dict
