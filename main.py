from app.llm import ask_llm

print("Testing LLMod.ai connection...")
response = ask_llm(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say the word 'Success' if you can hear me."
)
print(f"Result: {response}")