from .base_llm import BaseLLM

#used help from copilot

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
                    "You are a calculator that answers unit conversion questions. "
                    "Give a short explanation and then compute the result. "
                    "Always write your final answer like this:\n"
                    "<answer>NUMBER</answer>\n"
                    "Never include any text after </answer>. Just return the number."
                )
            },
            {
                "role": "user",
                "content": "How many bits are in 2 kilobytes?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilobyte = 1024 bytes, and 1 byte = 8 bits. Equation: 2 × 1024 × 8 = <answer>16384</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 3 kilograms to pounds."
            },
            {
                "role": "assistant",
                "content": (
                    "1 kilogram = 2.20462 pounds. Equation: 3 × 2.20462 = <answer>6.61386</answer>"
                )
            },
            {
                "role": "user",
                "content": "How many weeks are in 4 years?"
            },
            {
                "role": "assistant",
                "content": (
                    "1 year = 365.25 days, and 1 week = 7 days. Equation: (4 × 365.25) ÷ 7 = <answer>208.709827875</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 5 gallons to fluid ounces."
            },
            {
                "role": "assistant",
                "content": (
                    "1 gallon = 128 fluid ounces. Equation: 5 × 128 = <answer>640</answer>"
                )
            },
            {
                "role": "user",
                "content": "Convert 3 miles per hour to meters per second."
            },
            {
                "role": "assistant",
                "content": (
                    "1 mi/h = 0.44704 m/s. Equation: 3 × 0.44704 = <answer>1.34112</answer>"
                )
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