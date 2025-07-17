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
                "content": (
                    "You are an expert in unit conversions. "
                    "briefly summarize the question and then respond with the <answer>answer</answer> wrapped in <answer>NUMBER</answer>. "
                    # "Always provide a short step-by-step explanation, "
                    # "and end your response with only the final number wrapped in <answer> tags, like this: <answer>24</answer>. "
                    # "Do not include anything after the <answer> tag. This tag is required for correct parsing."
            )
            },
            {
                "role": "user",
                "content": "Convert 2 yards to feet."
            },
            {
                "role": "assistant",
                "content": "There is 3 feet in a yard. Two times three is <answer>6</answer>"
            },
            {
                "role": "user",
                "content": "How many seconds are there in a 3 minutes"
            },
            {
                "role": "assistant",
                "content": "There are 60 seconds in a minute. 3 times sixty is <answer>360</answer>"
            },
            # {
            #     "role": "user",
            #     "content": "Convert 20 yards to feet."
            # },
            # {
            #     "role": "assistant",
            #     "content": "One yard is 3 feet. Twenty times 3 is <answer>60</answer> feet."
            # },
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
    # testset = ["What does 2 liter equal in millilitre terms?"]
    # model = CoTModel()
    # for t in testset:
    #     print("testing answer function")
    #     print("input", t)
    #     answer = model.answer(t)
    #     print("final output is", answer)
    #answers = model.batched_generate(testset)
    #print(answers)

if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})