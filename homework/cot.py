from .base_llm import BaseLLM


class CoTModel(BaseLLM):

    def __init__(self):
        super().__init__()

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful and concise expert at converting units of measurement. Provide only the answer, with the final value wrapped in <answer> tags."
            },
            {
                "role": "user",
                "content": "Convert 5 kilometers to meters."
            },
            {
                "role": "assistant",
                "content": "5 kilometers is 5000 meters. <answer>5000</answer>"
            },
            {
                "role": "user",
                "content": "Convert 2 hours to minutes."
            },
            {
                "role": "assistant",
                "content": "2 hours is 120 minutes. <answer>120</answer>"
            },
            {
                "role": "user",
                "content": "Convert 3 feet to inches."
            },
            {
                "role": "assistant",
                "content": "3 feet is 36 inches. <answer>36</answer>"
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