from typing import Callable, Optional
from mlx_lm import load, generate
import mlx.nn as nn


def main():
    print("Loading model...")
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-MLXTuned")
    print("Model loaded! Enter your prompts below (/q to quit)")

    # Forward pass to capture layer 16 activations
    activations = []

    def hook_fn(outputs):
        print(outputs.shape)
        activations.append(outputs.detach())

    original_fn = model.layers[len(model.layers) - 1].__call__

    def hook_fn_wrapper(*args, **kwargs):
        outputs = original_fn(*args, **kwargs)
        hook_fn(outputs)
        return outputs

    model.layers[len(model.layers) - 1].__call__ = hook_fn_wrapper

    while True:
        try:
            prompt = input("\nPrompt> ")

            if prompt.strip() == "/q":
                print("Goodbye!")
                break

            if (
                hasattr(tokenizer, "apply_chat_template")
                and tokenizer.chat_template is not None
            ):
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            print("\nGenerating response...")
            response = generate(model, tokenizer, prompt=prompt, verbose=False)
            print("\nResponse:", response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
