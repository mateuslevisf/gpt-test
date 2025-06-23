### GPT Test

This repo holds files for testing the OpenAI API.

Before installing the packages, it is suggested that the user creats a conda environment:
```
conda create -n openai_test python=3.11
```

Then, to activate the environment:
```
conda activate openai_test
```

And finally:
```
pip install -r requirements.txt
```

To run the basic conversation client:
```
python scripts/conversation.py
```

When conversing with ChatGPT, you can write "save [filename]" to save your history or "quit" to quit conversing.

To run the RAG/Full-Context/Default LLM comparison:
```
python scripts/comparison.py
```

To test GPT next-token prediction:
```
python scripts/gpt.py
```