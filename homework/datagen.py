from .cot import CoTModel
import json
import re
from .data import is_answer_valid

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.2, batch_size: int = 16):
    from .data import Dataset
    from more_itertools import chunked  # optional: pip install more-itertools

    testset = Dataset("train")
    print("Dataset size:", len(testset))

    model = CoTModel(include_raw_response=True)
    gen_data = []

    # Chunk testset into batches
    for batch_idx, batch in enumerate(chunked(testset, batch_size)):
        print(f"\nProcessing batch {batch_idx + 1}...")

        questions = [q for q, _ in batch]
        gold_answers = [a for _, a in batch]

        prompts = [model.format_prompt(q) for q in questions]
        all_generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

        # all_generations is a list of lists: [ [sample1, sample2, ..., sampleN], ... ]
        for i, generations in enumerate(all_generations):
            found = False
            for sample in generations:
                parsed = model.parse_answer(sample)
                if is_answer_valid(parsed, gold_answers[i]):
                    block = extract_last_answer_block(sample)
                    if block:
                        gen_data.append((questions[i], gold_answers[i], block))
                        found = True
                        break  # Stop at first valid match
            if not found:
                # Optionally collect unmatched items for debugging
                pass

        with open(output_json + ".json", "w") as f:
            print(f"Writing {len(gen_data)} items to {output_json}.json")
            json.dump(gen_data, f)

def generate_dataset_single(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .data import Dataset, benchmark

    testset = Dataset("train")

    print("Dataset size:", len(testset))
    #testset = testset[:5]

    model = CoTModel(include_raw_response=True)
    gen_data = []

    counter = 1

    for question,answer in testset:

        print("Processing item number:", counter)

        question_input = [model.format_prompt(question)]
        generations = model.batched_generate(question_input, num_return_sequences=oversample, temperature= .1)

        for sample in generations:
            
            answer_response = model.parse_answer(sample[0])

            if answer_response == answer:

                last_block = extract_last_answer_block(sample[0])
                gen_data.append((question, answer, last_block))
                break  # break out of the loop once a correct answer is found

        counter += 1

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
    block_matches = list(re.finditer(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", text, flags=re.DOTALL))

    # Find the block that contains the last answer
    for match in reversed(block_matches):  # reverse to find the latest block containing the answer
        if match.start() <= last_answer_pos <= match.end():
            return match.group(1).strip()

    return None




if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
