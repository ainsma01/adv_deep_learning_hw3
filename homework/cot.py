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
                        "You are an expert in unit conversions. For each question, provide a concise response that includes: "
                        "1. A brief summary of the conversion requested. "
                        "2. A clear, step-by-step explanation of the conversion process. "
                        "3. The final numerical answer wrapped in <answer>NUMBER</answer> tags, with no units or additional text after the tag. "
                        "Use standard conversion factors (e.g., 1 yard = 3 feet, 1 minute = 60 seconds, 1 metric ton = 1000 kilograms, 1 megabyte = 1000 kilobytes, 1 US ton = 2000 pounds). "
                        "If the unit is ambiguous (e.g., 'ton'), assume US ton unless specified as 'metric ton'. "
                        "If the question is invalid or unclear, respond with an explanation and <answer>0</answer>."
                    )
                },
                {
                    "role": "user",
                    "content": "Convert 2 yards to feet."
                },
                {
                    "role": "assistant",
                    "content": "The question asks for 2 yards to be converted to feet. There are 3 feet in a yard. 2 times 3 is <answer>6</answer>"
                },
                {
                    "role": "user",
                    "content": "How many seconds are there in 3 minutes?"
                },
                {
                    "role": "assistant",
                    "content": "The question asks for 3 minutes to be converted to seconds. There are 60 seconds in a minute. 3 times 60 is <answer>180</answer>"
                },
                {
                    "role": "user",
                    "content": "Convert 5 kilograms to grams."
                },
                {
                    "role": "assistant",
                    "content": "The question asks for 5 kilograms to be converted to grams. There are 1000 grams in a kilogram. 5 times 1000 is <answer>5000</answer>"
                },
                {
                    "role": "user",
                    "content": "How many megabytes are in 4 gigabytes?"
                },
                {
                    "role": "assistant",
                    "content": "The question asks for 4 gigabytes to be converted to megabytes. There are 1000 megabytes in a gigabyte. 4 times 1000 is <answer>4000</answer>"
                },
                {
                    "role": "user",
                    "content": "Convert 6 pounds to ounces."
                },
                {
                    "role": "assistant",
                    "content": "The question asks for 6 pounds to be converted to ounces. There are 16 ounces in a pound. 6 times 16 is <answer>96</answer>"
                },
                {
                    "role": "user",
                    "content": "What is 3 tons in pounds?"
                },
                {
                    "role": "assistant",
                    "content": "The question asks for 3 tons to be converted to pounds. Assuming US tons, there are 2000 pounds in a ton. 3 times 2000 is <answer>6000</answer>"
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