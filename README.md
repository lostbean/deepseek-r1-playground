# DeepSeek R1 - Playground

# Generating training and validation data for fine-tuning

```bash
poetry run python ./generate.py data/questions.json data/train.jsonl data/valid.jsonl 0.2
```

This will use a local LLM to generate artificial train data from a local Ollama model. The initial set of questions is located in `data/questions.json`
