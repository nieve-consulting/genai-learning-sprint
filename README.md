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

Generative AI use cases, project lifecycle, and model pre-training

#### Lab 1 Learning Objectives

- Discuss model pre-training and the value of continued pre-training vs fine-tuning
- Define the terms Generative AI, large language models, prompt, and describe the transformer architecture that powers LLMs
- Describe the steps in a typical LLM-based, generative AI model lifecycle and discuss the constraining factors that drive decisions at each step of model lifecycle
- Discuss computational challenges during model pre-training and determine how to efficiently reduce memory footprint
- Define the term scaling law and describe the laws that have been discovered for LLMs related to training dataset size, compute budget, inference requirements, and other factors

### Lab 2

Fine-tuning and evaluating large language models

#### Lab 2 Learning Objectives

- Describe how fine-tuning with instructions using prompt datasets can improve performance on one or more tasks
- Define catastrophic forgetting and explain techniques that can be used to overcome it
- Define the term Parameter-efficient Fine Tuning (PEFT)
- Explain how PEFT decreases computational cost and overcomes catastrophic forgetting
- Explain how fine-tuning with instructions using prompt datasets can increase LLM performance on one or more tasks

### Lab 3

Building a RAG (Retrieval Augmented Generation) system.

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
