from .cot import CoTModel
import json

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .data import Dataset, benchmark

    testset = Dataset("train")

    testset = testset[:5]

    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_data = []

    for question,answer in testset:

        print("Question: ", question)
        print("Answer: ", answer)

        #generating samples
        samples = model.batched_generate(question, num_return_sequences=oversample, temperature=temperature)

        for sample in samples:

            #checking correctness of current sample
            print("Current sample: ", sample)
            if model.parse_answer(sample) == answer:

                #found correct answer appending it to the dataset
                gen_data.append(f'{question}, {answer}, {sample}')
                continue

    print("Generated dataet: ", gen_data)
    with open(output_json + ".json", "w") as f:
        json.dump(gen_data, f)



if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
