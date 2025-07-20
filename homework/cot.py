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
                    "You are a precise unit converter. For each question, explain the conversion briefly, "
                    "then wrap the final numeric result in <answer> tags. "
                    "Do not include any text after the closing </answer> tag."
                )
            },
            {
                "role": "user",
                "content": "How many bits are in 2 kilobytes?"
            },
            {
                "role": "assistant",
                "content": "1 kilobyte is 1024 bytes, and 1 byte is 8 bits. 2 × 1024 × 8 = <answer>16384</answer>"
            },
            {
                "role": "user",
                "content": "Convert 3 kilograms to grams."
            },
            {
                "role": "assistant",
                "content": "1 kilogram is 1000 grams. 3 × 1000 = <answer>3000</answer>"
            },
            {
                "role": "user",
                "content": "What is the conversion from 5 gallons to fluid ounces?"
            },
            {
                "role": "assistant",
                "content": "1 gallon is 128 fluid ounces. 5 × 128 = <answer>640</answer>"
            },
            {
                "role": "user",
                "content": "Please convert 3 miles per hour to meters per second."
            },
            {
                "role": "assistant",
                "content": "1 mi/h is about 0.44704 m/s. 3 × 0.44704 = <answer>1.34112</answer>"
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