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
                    "Always provide your reasoning steps in full sentences before giving the final answer in the <answer>...</answer> tag."
                )
            },
            {
                "role": "user",
                "content": "Convert 10 inches to centimeters."
            },
            {
                "role": "assistant",
                "content": "There are 2.54 centimeters in an inch. Ten times 2.54 is <answer>25.4</answer>"
            },
            {
                "role": "user",
                "content": "How many grams are there in 5 kilograms?"
            },
            {
                "role": "assistant",
                "content": "There are 1000 grams in a kilogram. Five times 1000 is <answer>5000</answer>"
            },

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

    print(model.include_raw_response)

    testset = testset[:3]

    for question,answer in testset:

        question_input = [model.format_prompt(question)]
        generations = model.batched_generate(question_input, num_return_sequences=1, temperature=0.1)
        print("Question is:", question)
        print("Generations is:", generations[0][0])
        answer = model.parse_answer(generations[0][0])

        print("output", answer)

        # formatted_question = model.format_prompt(question)
        # print("testing answer function")
        # print("input", formatted_question)
        # raw_answer = model.batched_generate(formatted_question)
        # answer = model.parse_answer(raw_answer)
        # print("Raw output is:", raw_answer)
        # print("Answer is:", answer)

        # print("Now trying by just calling answer")
        # answer_response = model.answer(question)
        # print("Answer is:", answer_response)
        # print("Answer function response is:", answer_response[0][0])
        # print("Raw output is:", answer_response[0][1])

    # benchmark_result = benchmark(model, testset, 100)
    # print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")
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