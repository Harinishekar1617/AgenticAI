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

**Reinforcement learning**

learning to directly optimize something with human feedback
3 core components
- pretraining a language model
- gathering data and training a reward model
  <img width="1400" height="1046" alt="image" src="https://github.com/user-attachments/assets/79bfadf3-03cd-4021-bf79-0a0a3f19c5fc" />

PASSING prompts to the base model and then generating a scalar reward with human preferences to build a reward model
- fine tuning LM with RL
**Example: Training a Reward Model for a Chatbot**
Goal: Train a chatbot to generate helpful and polite responses.

1. Data Collection:

Prompt: "What is the capital of France?"

Generate Multiple Responses (e.g., from an initial LLM):

Response A: "Paris is the capital." (Helpful, concise)

Response B: "The capital of France is Paris, of course. Didn't you know that?" (Helpful but rude)

Response C: "I like dogs." (Irrelevant)

Response D: "The capital of France is Paris." (Helpful, polite)

Human Annotation: A human annotator receives these responses and is asked to rank them.

Human's judgment: D > A > B > C (D is best, then A, then B, then C)

From this judgment, we create "reward pairs":

(Prompt, D) vs (Prompt, A) (D is preferred over A)

(Prompt, A) vs (Prompt, B) (A is preferred over B)

(Prompt, B) vs (Prompt, C) (B is preferred over C)

...and all other possible pairs.

**Reward Model Training:**
Model Input: A pair of (prompt, response) combinations.
Model Output: A single scalar score for each (prompt, response) combination.
Training Loop:
Take a batch of human-labeled preference pairs (e.g., (Prompt, D) as "chosen" and (Prompt, B) as "rejected").

**After Training:**

The trained reward model can now take any new (prompt, response) pair and predict a numerical score that approximates how a human would rate that response. This score then becomes the "reward signal" that is fed to the RL agent (the main chatbot model) during its reinforcement learning phase (e.g., using PPO). The RL agent then learns to generate responses that maximize this predicted reward, thus aligning its behavior with human preferences without needing constant human supervision during its core training loop.

<img width="2080" height="1571" alt="image" src="https://github.com/user-attachments/assets/9d15a5e4-25e9-4bdc-ac94-8bc4de3cd5fc" />

**Simple Explanation: Policy is more like a rule book that exists for the RL agent**
A policy answers the question: "Given what I see now (my current state), what should I do next (what action should I take)?"

More Technical Explanation:
Mathematically, a policy is usually denoted by π (pi). It's a mapping from states to actions:

Deterministic Policy: For every state s, the policy π(s) directly tells you one specific action a to take.
Example: "If I am in state S1, always take action A 3
​Stochastic Policy: For every state s, the policy π(a∣s) gives you a probability distribution over all possible actions. This means it tells you the likelihood of taking each action from that state.

Example: "If I am in state S 1take action A 1with 70% probability, and action A 2with 30% probability." Stochastic policies are often preferred, especially during learning, as they allow for exploration (trying different actions to see what happens).

The action space is all the actions the agent can take, in this context it ids all the tokens that are available in the voxabulary of large language model
the observation space is all that the agent can observe

<img width="1973" height="1682" alt="image" src="https://github.com/user-attachments/assets/67f73994-bf78-4a42-ab03-700937d99df9" />
