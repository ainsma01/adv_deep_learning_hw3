from .base_llm import BaseLLM


class CoTModel(BaseLLM):

    def __init__(self, checkpoint='HuggingFaceTB/SmolLM2-360M-Instruct', include_raw_response=False):
        super().__init__(checkpoint=checkpoint, include_raw_response=include_raw_response)

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in unit conversions. "
                    "Provide a step-by-step explanation and conclude with only the final number wrapped in <answer> tags, e.g. <answer>24</answer>. "
                    "Do not include any text after the <answer> tag."
                )
            },
            {
                "role": "user",
                "content": "Convert 5 kilograms to grams."
            },
            {
                "role": "assistant",
                "content": "There are 1000 grams in a kilogram. Five times 1000 is <answer>5000</answer>"
            },
            {
                "role": "user",
                "content": "Convert 3 meters to centimeters."
            },
            {
                "role": "assistant",
                "content": "There are 100 centimeters in a meter. Three times 100 is <answer>300</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]




        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return prompt

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()

    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})