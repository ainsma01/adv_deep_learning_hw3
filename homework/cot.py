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
                    "You are a helpful assistant that performs unit conversions. "
                    "Always wrap the final numeric result in <answer> tags. "
                    "Explain your reasoning briefly, then provide the final value like this: <answer>VALUE</answer>. "
                    "Do not include anything after the closing </answer>."
                )
            },
            {
                "role": "user",
                "content": "How many seconds are in 2.5 minutes?"
            },
            {
                "role": "assistant",
                "content": "There are 60 seconds in a minute. 2.5 times 60 is <answer>150</answer>"
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