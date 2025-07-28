**Langchain setup and Demo**

Install uv
Virtual environment setup
new working directory

uv venv --python 3.11.8
uv python pin 3.11.8
.venv\Scripts\activate

uv pip install langchain==0.3.25 langgraph==0.4.5 langchain-openai==0.3.18 python-dotenv==1.1.0
python -c "import langchain; print(langchain.__version__)"

**Generate OPENAI APi key**

Set up IPykernel for Jupyter notebook

uv pip install ipykernel jupyter
python -m ipykernel install --user --name=venv --display-name "Python (langchain)"

Run langchain_demo.ipynb file

-- Understanding here
Langchain : Core Purpose: A flexible, modular framework designed to simplify the creation of applications powered by LLMs.
langgraph : Core Purpose: An extension of LangChain specifically designed for building stateful, multi-actor 
(agentic) applications with LLMs using a graph-based approach.

Choose LangChain if your application involves a clear, sequential flow of LLM operations (e.g., simple Q&A, content generation, basic summarization). It's quicker to get started for these use cases.

Choose LangGraph if you need to build more sophisticated, autonomous, and stateful agents that require:

- Iterative reasoning (loops)
- Complex decision-making and branching
- Human intervention points
- Maintaining a shared, evolving state over many turns
- Multi-agent coordination

Runnables are the standard, connectable building blocks of LangChain that let you easily design and run complex AI workflows. Prompttempate, output parser
etc.. are all runnables

Message types
user message : how to represent inout from a human user
AI message'response from the AI model itslef
system message: instruct AI model on how to behave

Chaining
Chaining is the core idea behind how langchain works, it connects all the runnables together and produces an output. connects steps together.

**NotebookLMs epic use cases**
https://www.youtube.com/watch?v=9xjmvUS-UGU

Vercel's VP of Product on How to Use v0 to Build Your Own Ideas (Step-by-Step Guide)
Build a Full-Stack App in 7 Minutes with v0 (Figma to Code)

**Benchmarking Models**
chatbot arena
Start with medium sized models or mid priced models like GPT 4o and sonnet 3.5

** Generative AI terms **

<img width="2048" height="1404" alt="image" src="https://github.com/user-attachments/assets/5b29097f-a37f-425d-8256-27926e2e36ae" />

**Agentic AI terms **

<img width="2048" height="1404" alt="image" src="https://github.com/user-attachments/assets/ff635b53-7e96-40fb-95e9-1eaecc708b7b" />

**Our Favorite Resources on AI in the Enterprise (Read Them to Identify Use-Cases)**

https://www.zenml.io/llmops-database

https://cloud.google.com/transform/101-real-world-generative-ai-use-cases-from-industry-leaders

https://aws.amazon.com/ai/generative-ai/use-cases/?awsm.page-customer-references=1

**Shorter Learning Roadmap (2-3 hours)**

If you just want a quick, practical understanding of LLMs and transformers without diving into machine learning and deep learning, here are two great resources:

Jay Alammar’s Illustrated Transformers – https://jalammar.github.io/illustrated-transformer/

Andrej Karpathy’s video on LLM training – https://www.youtube.com/watch?v=7xTGNNLPyMI

Illustrating Reinforcement Learning from Human Feedback (RLHF) - https://huggingface.co/blog/rlhf

