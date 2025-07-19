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

def format_example(prompt: str, answer: str, summary: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    answer_float = float(answer)
    
    return {
        "question": prompt.strip(),
        "answer": f"<answer>{round(answer_float, 2)}</answer>"
    }

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
    dataset = Dataset("rft")
    
    # 4. Tokenize dataset
    tokenized_data = TokenizedDataset(tokenizer, dataset, format_example)

    #define training args
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=5e-4,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=16,
    )

    #call trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # 7. Train
    trainer.train()

    #save model
    trainer.save_model("homework/sft_model")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
