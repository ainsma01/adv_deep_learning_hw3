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
                    "only answer in the format <answer>NUMBER</answer>. "
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
                "content": "<answer>6</answer>"
            },
            # {
            #     "role": "user",
            #     "content": "Convert 2 yards to feet."
            # },
            # {
            #     "role": "assistant",
            #     "content": "One yard is 3 feet. Two times 3 is <answer>6</answer> feet."
            # },
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
    # testset = ["How many feet in 20 yards?", "How many hours in a day?"]
    # model = CoTModel()
    # for t in testset:
    #     print("testing answer function")
    #     print("input", t)
    #     answer = model.answer(t)
    #     #rint("output", answer)
    #answers = model.batched_generate(testset)
    #print(answers)

if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})