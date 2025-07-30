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

**Transformers**

- broken down into encoders and decoders
- Encoding component consists of feed forward neural network and self-attention layer
  Encoders first inout flow through self attention layer,a layer that helps look at other words in the sentence as it encodes a specific word. These outputs are fed to feed forward neural network
- Decoder has both layers,but between them is attention that helps the decoder focus on relevant parts of the input sentence

Encoding
<img width="1268" height="771" alt="image" src="https://github.com/user-attachments/assets/5f442b84-d079-4d8c-b1f6-b4f981074c06" />

Each word is represented as a vector with 512 dimensions and passed to self attention layer, then they pass through feed forward, the first level takes the vectors from word embeddings and the second layer takes the encoders from first layer
As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word.

**How self attention works**
The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word)

<img width="875" height="552" alt="image" src="https://github.com/user-attachments/assets/b76227e6-04d0-49be-be80-8daa83d2114c" />

The second step in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, “Thinking”. We need to score each word of the input sentence against this word

<img width="685" height="358" alt="image" src="https://github.com/user-attachments/assets/5df35f70-6791-4795-a87b-52052bc37350" />

The third and fourth steps are to divide the scores by 8 and passit through a softmax function so the scores are all positive and add upto 1
the fifth step is to multiply it with the value vector and then finally sum up that to arrive at the embedding output from self attention layer




