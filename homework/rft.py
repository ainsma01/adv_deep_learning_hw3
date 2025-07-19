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
    return {
        "question": prompt.strip(),
        "answer": summary.strip(),
    }

def train_model(
    output_dir: str,
    epochs = 20,
):
    #base model
    base_llm = BaseLLM()

    #create LORA
    rank = 16
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r = rank,
        lora_alpha = rank * 4
    )

    lora_model = get_peft_model(base_llm.model, lora_config)
    lora_model.enable_input_require_grads()

    # 3. Load dataset with reward fine-tuning format
    dataset = Dataset("rft")

    print("Dataset size:", len(dataset))
    
    # 4. Tokenize dataset
    tokenized_data = TokenizedDataset(base_llm.tokenizer, dataset, format_example)

    #define training args
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=1e-4,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
    )

    #call trainer
    from transformers import Trainer

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # 7. Train
    trainer.train()

    #save model
    trainer.save_model("homework/rft_model")

    #test model
    test_model("homework/rft_model")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
