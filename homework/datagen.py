from .cot import CoTModel
import json
import re

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .data import Dataset, benchmark

    testset = Dataset("valid")

    testset = testset[:3]

    model = CoTModel(include_raw_response=True)
    gen_data = []

    for question,answer in testset:

        question_input = [model.format_prompt(question)]
        generations = model.batched_generate(question_input, num_return_sequences=oversample, temperature=temperature)

        for sample in generations:
            if model.parse_answer(sample) == answer and last_block is not None:

                print("Holy shit we did it")
                last_block = extract_last_answer_block(sample)
                gen_data.append((question, answer, last_block))

       
            



    print("Generated dataet: ", gen_data)
    with open(output_json + ".json", "w") as f:
        json.dump(gen_data, f)

def extract_last_answer_block(text: str) -> str | None:
    """
    Finds the last <answer>...</answer> tag and returns the enclosing <|im_start|> ... <|im_end|> block that contains it.
    If no answer tag is found, returns None.
    """
    # Find the last occurrence of <answer>...</answer>
    answer_match = list(re.finditer(r"<answer>.*?</answer>", text, flags=re.DOTALL))
    if not answer_match:
        return None
    
    last_answer_pos = answer_match[-1].start()

    # Find all <|im_start|> ... <|im_end|> blocks with their positions
    block_matches = list(re.finditer(r"<\|im_start\|>(.*?)<\|im_end\|>", text, flags=re.DOTALL))

    # Find the block that contains the last answer
    for match in reversed(block_matches):  # reverse to find the latest block containing the answer
        if match.start() <= last_answer_pos <= match.end():
            return match.group(0).strip()

    return None




if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
