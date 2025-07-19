from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset
from .data import Dataset, benchmark
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from .base_llm import BaseLLM
from .data import Dataset

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    #base model
    # 1. Load base model
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer

    # 2. Enable LoRA adapter
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load dataset with reward fine-tuning format
    dataset = Dataset("homework/rft_dataset.jsonl")
    raw_data = dataset.examples  # [(question, correct_answer, reasoning_with_answer_tag), ...]

    # 4. Format for training: question -> CoT reasoning
    def format_rft_example(example):
        question, _, reasoning = example
        return {
            "input": question.strip(),
            "output": reasoning.strip(),
        }

    formatted_data = list(map(format_rft_example, raw_data))

    # 5. Tokenize
    def tokenize(example):
        prompt = f"{example['input'].strip()}\n"
        target = f"{example['output'].strip()}"
        full = tokenizer(prompt + target, padding="max_length", truncation=True, max_length=256)
        full["labels"] = full["input_ids"].copy()
        return full

    tokenized = list(map(tokenize, formatted_data))
    tokenized_dataset = TokenizedDataset(tokenized)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=5,
            logging_steps=1,
            save_steps=250,
            save_total_limit=1,
            learning_rate=2e-4,
            bf16=torch.cuda.is_bf16_supported(),
            report_to=[],
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 7. Train
    trainer.train()

    #save model
    trainer.save_model("homework/sft_model")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
