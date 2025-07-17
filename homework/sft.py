from .base_llm import BaseLLM
from .data import Dataset, benchmark
from peft import LoraConfig, get_peft_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    answer_float = float(answer)
    
    return {
        "question": prompt,
        "answer": f"<answer>{answer_float}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    #base model
    base_llm = BaseLLM()

    #create LORA
    rank = 8
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r = rank,
        lora_alpha = rank * 4
    )

    lora_model = get_peft_model(base_llm.model, lora_config)
    lora_model.enable_input_require_grads()

    #get training data
    train_data = Dataset("train")
    tokenized_data = TokenizedDataset(base_llm.tokenizer, train_data, format_example)

    #define training args
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=2e-4,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=32,
    )

    #call trainer
    from transformers import Trainer

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()

    #save model
    trainer.save_model("homework/sft_model")

    


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
