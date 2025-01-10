from unsloth import FastLanguageModel

if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained("runs/run5/checkpoint-2000")
    model = FastLanguageModel.for_inference(model)

    prompt = """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    peft_config = LoraConfig(
    """

    outputs = model.generate(
        input_ids=tokenizer(prompt, return_tensors="pt").input_ids.cuda(),
        max_new_tokens=128,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(response)