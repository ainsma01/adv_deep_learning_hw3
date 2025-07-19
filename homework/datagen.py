from .cot import CoTModel
import json

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .data import Dataset, benchmark

    testset = Dataset("train")

    testset = testset[:5]

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_data = []

    for question,answer in testset:
        samples = model.batched_generate(question, num_return_sequences=oversample, temperature=temperature)

        for sample in samples:
            if model.parse_answer(sample) == answer:
                gen_data.append(f'{question}, {answer}, {sample}')
                continue
    open(output_json + ".json", "w")
    with open(output_json + ".json", "w") as f:
        json.dump(gen_data, f)



if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
