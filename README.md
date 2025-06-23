### GPT Test

This repo holds files for testing the OpenAI API.

Before installing the packages, it is suggested that the user create a conda environment (must have miniconda or Anaconda installed):
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

To use scripts that call the OpenAI API (most in the ``scripts`` folder except ```pdf_processor.py``` and ```gpt.py```), you must first create a ```.env``` file in the root directory of this project and add a ```OPENAI_API_KEY=[key]``` line, where ```[key]``` is a working OpenAI key.

To run the basic conversation client:
```
python scripts/conversation.py
```

When conversing with ChatGPT, you can write "save [filename]" to save your history or "quit" to quit conversing.

To run the RAG/Full-Context/Default LLM comparison:
```
python scripts/comparison.py
```

Note that for better use of the comparison script, it is recommended that the user loads some PDF file. The "rio_wikipedia.pdf" file can be used for that, but then it is recommended that questions center around Rio de Janeiro; to use another pdf file, just save it inside the project folder.

To test GPT next-token prediction (uses GPT-2 in memory):
```
python scripts/gpt.py
```