import os
import random
import re

import huggingface_hub
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import load_dataset
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

import wandb

# Note
# This is mostly an adaptation this [simple experimentation](https://github.com/emailtovamos/DeepSeekR1Zero) to use MLX and run on Apple Silicon
# Also used some is inspiration [RLX](https://github.com/noahfarr/rlx)


wandb.init(project="DeepSeek-R1-Zero-Training")

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
mx.random.seed(SEED)

# Load dataset
dataset = load_dataset("gsm8k", "main", split="train[:20]")


def preprocess_dataset(dataset):
    """Add format instructions to each example as per DeepSeek R1 Zero paper"""
    processed_data = []
    for sample in dataset:
        # Create system prompt as shown in the paper
        system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: """

        question = sample["question"]
        answer = sample["answer"]
        numeric_answer = answer.split("####")[-1].strip()

        # Combine system prompt with question as per paper's template
        full_prompt = f"{system_prompt}{question}\nAssistant:"

        # Format expected response with think and answer tags
        formatted_answer = f"<think>{answer}</think>\n<answer>{numeric_answer}</answer>"

        processed_data.append(
            {
                "prompt": full_prompt,
                "response": formatted_answer,
                "original_question": question,
                "expected_answer": numeric_answer,
            }
        )

    return processed_data


def compute_reward(response_text, expected_answer):
    """Enhanced reward function based on DeepSeek R1 Zero paper"""
    reward = 0.0

    # Check for presence of think tags
    has_think_start = "<think>" in response_text.lower()
    has_think_end = "</think>" in response_text.lower()

    # Check for presence of answer tags
    has_answer_start = "<answer>" in response_text.lower()
    has_answer_end = "</answer>" in response_text.lower()

    # Format reward (0.4 total for format)
    if has_think_start and has_think_end:
        reward += 0.2  # Reward for using think tags
    if has_answer_start and has_answer_end:
        reward += 0.2  # Reward for using answer tags

    # Extract answer and check accuracy
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.search(answer_pattern, response_text, flags=re.IGNORECASE | re.DOTALL)

    if match:
        extracted_answer = match.group(1).strip()
        expected_answer_cleaned = expected_answer.split("####")[-1].strip()

        # Accuracy reward (0.6 for correct answer)
        if extracted_answer == expected_answer_cleaned:
            reward += 0.6

    return reward


def evaluate_model(model, tokenizer, prompt, logging_prefix="", max_tokens=400):
    """Evaluate model on a single prompt with detailed logging"""

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generate(
        model, tokenizer, prompt=prompt, verbose=True, max_tokens=max_tokens
    )

    # Log the response
    print(f"\n{logging_prefix}")
    print("=" * 50)
    print("Input prompt:", prompt)
    print("-" * 50)
    print("Model response:", response)
    print("=" * 50)

    # Extract thinking process and answer if present
    think_match = re.search(
        r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE
    )
    answer_match = re.search(
        r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE
    )

    thinking = (
        think_match.group(1).strip() if think_match else "No thinking process found"
    )
    answer = answer_match.group(1).strip() if answer_match else "No answer found"

    # Log structured components
    print("\nStructured Analysis:")
    print("Thinking Process:", thinking)
    print("Final Answer:", answer)
    print("-" * 50)

    return response


class DeepSeekTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        lr=1e-6,
        epsilon=0.2,
        group_size=4,
        temp=0.5,
        max_tokens=400,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(learning_rate=lr)
        self.epsilon = epsilon
        self.group_size = group_size
        self.sampler = make_sampler(temp)
        self.max_tokens = max_tokens

        print("Loading reference model...")
        self.reference_model = load("Qwen/Qwen2.5-1.5B")
        print("Reference model loaded!")

    def train_step(self, dataset):
        total_loss = 0

        for batch_idx, sample in enumerate(tqdm(dataset, desc="Training...")):
            prompt = sample["prompt"]
            correct_answer = sample["response"]

            # Evaluate first sample in detail periodically
            if batch_idx == 0:
                evaluate_model(
                    self.model,
                    self.tokenizer,
                    prompt,
                    f"Current model response (during training batch {batch_idx})",
                    max_tokens=self.max_tokens,
                )

            # Training logicmessages = [{"role": "user", "content": prompt}]
            messages = [{"role": "user", "content": prompt}]
            generated_ids = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            old_samples, old_rewards = [], []

            for _ in range(self.group_size):
                generated_text = generate(
                    model,
                    tokenizer,
                    sampler=self.sampler,
                    prompt=generated_ids,
                    verbose=True,
                    max_tokens=self.max_tokens,
                )
                reward = compute_reward(generated_text, correct_answer)
                old_samples.append(generated_text)
                old_rewards.append(reward)

            mean_reward = np.mean(old_rewards)
            std_reward = np.std(old_rewards) if np.std(old_rewards) > 1e-6 else 1e-6
            advantages = [(r - mean_reward) / std_reward for r in old_rewards]

            def loss_fn(sample_id):
                # Get logits for the generated text
                tokens = tokenizer.encode(sample_id)
                logits = model(mx.array([tokens]))

                # Compute policy loss using PPO
                log_probs = nn.log_softmax(logits[0, :, :], axis=-1)
                return -mx.mean(
                    mx.minimum(
                        mx.exp(log_probs) * advantage,
                        mx.clip(mx.exp(log_probs), 1 - self.epsilon, 1 + self.epsilon)
                        * advantage,
                    )
                )

            for sample_ids, advantage in zip(old_samples, advantages):
                policy_loss, grads = nn.value_and_grad(model, loss_fn)(sample_ids)
                # Update model parameters
                self.optimizer.update(model, grads)

                total_loss += policy_loss
                wandb.log(
                    {
                        "loss": policy_loss.item(),
                        "reward": np.mean(old_rewards),
                        "advantage": advantage,
                    }
                )

        return total_loss / len(dataset)


if __name__ == "__main__":
    print("Starting DeepSeek-R1-Zero training...")

    # Initialize tokenizer and model
    access_token = os.environ["HF_TOKEN"]
    huggingface_hub.login(access_token)
    model, tokenizer = load("Qwen/Qwen2.5-1.5B")

    # Preprocess dataset
    processed_dataset = preprocess_dataset(dataset)

    # Get a sample question for consistent evaluation
    test_sample = processed_dataset[0]
    print("\nSelected test sample:")
    print("Question:", test_sample["original_question"])
    print("Expected Answer:", test_sample["expected_answer"])

    # Evaluate base model before any training
    print("\nEvaluating base model before training:")
    evaluate_model(
        model, tokenizer, test_sample["prompt"], "BASE MODEL (Before Training)"
    )

    # Initialize trainer
    trainer = DeepSeekTrainer(model, tokenizer, group_size=4, max_tokens=400)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")

        # Train
        loss = trainer.train_step(processed_dataset)
        print(f"Epoch {epoch + 1} Loss: {loss:.4f}")

        # Evaluate after epoch
        print(f"\nEvaluating after epoch {epoch + 1}:")
        evaluate_model(
            model, tokenizer, test_sample["prompt"], f"MODEL AFTER EPOCH {epoch + 1}"
        )

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/deepseek_r1_zero_epoch_{epoch + 1}"
            model.save_weights(save_path)
            print(f"Saved checkpoint to {save_path}")
