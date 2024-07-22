# GenAI Learning Sprint

In this learning sprint, you'll explore the fundamentals of Generative AI and Large Language Models (LLMs). You'll learn how to interact with LLMs, fine-tune them, and evaluate their performance. You'll also explore the risks and challenges associated with generative AI systems and learn how to build and evaluate generative AI systems.

## Outline

By the end of this learning sprint, you'll learn:

### 1. Understanding Generative AI Basics

- Definition and distinction from traditional AI
- Overview of applications: language, audio, image, video
- Brief look at key architectures: VAEs, GANs, diffusion models, Transformers
- Role of Large Language Models (LLMs) in the ecosystem
- Lab discussion: Pre-trained diffusion models and LLMs

### 2. Machine Learning Fundamentals

- Core principles of Machine Learning
- Discriminative vs. generative AI
- Learning paradigms: Supervised, Unsupervised, Reinforcement Learning
- Industry use case classification exercise
- Introduction to deep learning and neural networks
- Hands-on demo: Tensorflow Playground

### 3. Deep Dive into Large Language Models (LLMs)

- Definition, architecture, and role of Transformers in LLMs
- Autoregressive models and tokenization
- Capabilities and limitations of LLMs
- Foundation models: commercial vs. open-source
- Performance evaluation of LLMs
- Text generation: Sampling, decoding, output control

### 4. Interacting with LLMs

- Concept and techniques of prompting
- Prompt engineering: Zero-shot, one-shot, few-shot, chain-of-thought
- Advanced prompting techniques overview

### 5. Risks and Challenges in Generative AI Systems

- Hallucinations, biases, quality, and reliability concerns
- Model output comparison exercises
- Jailbreaking and guardrails demo
- Inference costs and latency measurement

### 6. Architecture of Generative AI Systems

- Common use cases and industry applications
- Customization and fine-tuning techniques
- Retrieval-Augmented Generation (RAG)
- Parameter-efficient fine-tuning, DPO, RLHF, RLAIF
- Defensive UX and user feedback integration
- Scalability, deployment, pruning, quantization, caching, batching exercises

### 7. Evaluating LLM Systems

- Evaluation metrics and benchmarks: ROUGE, BERT
- LLM-as-judge techniques
- Monitoring, auditing, interpretation, and debugging of LLM outputs

## Labs

### Lab 1

Prompt engineering and effective prompt design

#### Lab 1 Learning Objectives

1. Understand the fundamental principles of prompt engineering, including how to craft effective and clear prompts for AI models.
2. Explore the role of linguistics in prompt engineering, emphasizing the importance of language nuances, grammar, and structure in creating effective prompts.
3. Learn about various types of prompts, such as discrete prompts, soft prompts, in-context learning, and few-shot learning, and how they influence AI responses.
4. Gain insights into designing effective prompts by analyzing the characteristics of good versus bad prompts, focusing on clarity, relevance, specificity, and balance.

### Lab 2

Instruction tuning and fine-tuning of LLMs

#### Lab 2 Learning Objectives

1. Understand the concept of instruction tuning and its significance in enhancing the performance of language models like LLaMA 2 by training them on input-output instruction pairs.
2. Learn the process of fine-tuning the LLaMA 2 model using a high-quality instruction dataset, including setting up the training environment, configuring model parameters, and utilizing specialized libraries and tools.
3. Apply the fine-tuned model to generate text based on specific prompts, demonstrating the model's ability to perform various instruction-driven tasks effectively.

### Lab 3

Resource Augmented Generation (RAG)

#### Lab 3 Learning Objectives

1. Understand the motivation behind Resource Augmented Generation (RAG) and how it enhances the capabilities of Large Language Models (LLMs) by providing domain-specific context and improving prediction accuracy.
2. Learn the advantages and disadvantages of RAG, including its ability to reduce hallucinations without increasing model parameters and the potential challenges such as increased latency and the need for data curation.
3. Gain hands-on experience with implementing RAG, including installing necessary modules, loading and chunking documents, creating embeddings, and setting up a vector database for efficient retrieval.
4. Explore advanced applications of RAG by integrating it with different frameworks and models, such as Llama 2, Langchain, and ChromaDB, to handle various data types and improve LLM responses.

## Before You Arrive ✅

> [!IMPORTANT]  
> Complete the following steps **BEFORE** you come to the tutorial
> 
> - [ ] [Google Colab](https://colab.research.google.com) **Setup Account**
> 
>     <details closed><summary><code>Google Colab Instructions</code></summary>
> 
>     The Colab platform gives the user a virtual machine in which to run Python codes including machine
>     learning codes.
> 
>     The VM comes with a preinstalled environment that includes most of what is needed
>     for these tutorials.
> 
>     * You need a Google Account to use Colaboratory
>     * Go to [Google's Colaboratory Platform](https://colab.research.google.com) and sign in with
>       your Google account
>     * Click on the `New Notebook` at the bottom
>     * Now you will see a new notebook where you can type in python code.
>     * After you enter code, type `<shift> + <enter>` to execute the code cell.
>     * A full introduction to the notebook environment is out of scope for this tutorial, but many
>       can be found with a [simple Google
>       search](https://www.google.com/search?q=jupyter+notebook+tutorial)
>     * We will be using notebooks from this repository during the tutorial, so  you should be
>       familiar with how to import them into Colaboratory
>     * Now you can open the `File` menu at the top left and select `Open Notebook` which will open a
>       dialogue box.
>     * Select the `GitHub` tab in the dialogue box.
>     * From here you can enter the url for the github repo
>       and hit `<enter>`.
>     * This will show you a list of the Notebooks available in the repo.
>     * As each session of the tutorial begins, you will simply select the corresponding notebook from
>       this list and it will create a copy for you in your Colaboratory account (all `*.ipynb` files in
>       the Colaboratory account will be stored in your Google Drive).
>     * To use a TPU, in the notbook the select `Runtime` -> `Change Runtime Type` and you have a
>       dropbox list of hardward settings to choose from where the notebook can run.
> 
>     * **Important:** in order to be able to make changes to the notebooks, you need to make a copy of the notebook that will be stored in your Google Drive account. In order to do so, click on the `Copy to Drive` button, located in the main editor bar, next to the `Code` and `Text` buttons.
>     </details>
> 
> 
> - [ ] 🤗 [Hugging Face](https://huggingface.co): **Account and Access Token**
> 
>     <details closed><summary><code>Hugging Face Instructions</code></summary>
> 
>     - Sign up for a huggingface account and obtain an access token: https://huggingface.co
>     - Sign Up (top bar)
>       Log into huggingface and get an access token:
>         - Login -> Settings (left panel) -> Access Tokens (left pane) -> New token (center pane)
> 
>     </details>
> 
> - [ ] 🦙 [Request access](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) to Llama-2 model
> 
>     <details closed><summary><code>Llama-2 Access Instructions</code></summary>
> 
>     - Visit this https://huggingface.co/meta-llama/Llama-2-7b-hf and request access to the model
>     - vist meta website and accept the terms https://ai.meta.com/resources/models-and-libraries/llama-downloads/
>     - Note: Your Hugging Face account email address MUST match the email you provide on the Meta website, or your request will not be approved.
> 
>     </details>


## External Resources

- [Generative AI Handbook](https://genai-handbook.github.io/)
