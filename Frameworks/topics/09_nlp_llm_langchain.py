"""
LangChain — The Framework for LLM-Powered Applications
=======================================================

LangChain is an open-source framework created by Harrison Chase in October 2022
and released publicly in November 2022. It became the fastest-growing open-source
project in GitHub history, reaching 10,000 stars in its first month. By 2023 it
had over $10M in seed funding and 200+ contributors. By 2025 it is the de-facto
standard framework for building applications on top of large language models.

The fundamental problem LangChain solves: large language models are powerful
but stateless, forgetful, and disconnected. They can reason and generate text,
but by themselves they cannot remember past conversations, cannot look up
current facts, cannot call external APIs, cannot execute code, cannot search
the web. They live in an isolated bubble of their training data.

LangChain provides the connective tissue that turns a raw LLM call into a
complete, stateful, tool-using application:

    RAW LLM:        prompt → model → text
    LANGCHAIN APP:  memory + retrieval + tools + prompt → model → structured
                    output → post-processing → action

The framework is built around a few powerful abstractions:
    - Models: unified interface to any LLM or chat model
    - Prompts: templated, versioned, reusable prompt construction
    - Chains: composable sequences of LLM calls and processing steps
    - Agents: LLMs that autonomously choose and use tools to achieve goals
    - Memory: stateful conversation history across multiple turns
    - Retrievers: systems that fetch relevant context from knowledge bases
    - LCEL: LangChain Expression Language — the composition DSL

LangChain 0.2+ introduced a cleaner architecture split across:
    langchain-core:     base abstractions (Runnable, BaseMessage, etc.)
    langchain:          chains, agents, memory implementations
    langchain-community: 600+ third-party integrations
    langchain-openai:   OpenAI-specific wrappers
    langchain-anthropic: Anthropic (Claude) wrappers
    langgraph:          stateful agent graphs (the modern agent framework)
    langserve:          deploy chains as REST APIs

This module covers the complete LangChain stack: the Runnable protocol and
LCEL composition, prompt engineering patterns, model abstraction layer, memory
systems, Retrieval Augmented Generation (RAG) from first principles, the agent
loop and tool use, LangSmith observability, and production deployment patterns.

"""

import textwrap
import re

TOPIC_NAME   = "LangChain — The Framework for LLM-Powered Applications"
DISPLAY_NAME = "09 · LangChain"
ICON         = "🦜"
SUBTITLE     = "From Prompt Templates to Production RAG Agents"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY LANGCHAIN EXISTS: THE LLM APPLICATION PROBLEM

### The Gap Between a Model and an Application

    A large language model, taken alone, has three fundamental limitations
    that prevent it from being deployed as a useful application:

    1. CONTEXT WINDOW LIMIT:
        Every LLM has a maximum input length (context window): 4K tokens
        for early GPT-3, 128K for GPT-4o, 200K for Claude-3. Documents,
        codebases, knowledge bases, entire websites — these vastly exceed
        any context window. A model cannot reason about what it cannot see.

    2. KNOWLEDGE CUTOFF:
        Models are trained on data up to a cutoff date. They cannot answer
        questions about events after training, current stock prices, today's
        news, or your proprietary internal documentation. The model's knowledge
        is frozen at training time.

    3. ACTION LIMITATION:
        Models generate text. They cannot execute Python code, call REST APIs,
        query databases, send emails, browse the web, or write files — unless
        explicitly given the ability to do so through a tool-calling mechanism.

    These three limitations define the three core pillars of LangChain:
        Problem 1 → RETRIEVAL:  fetch only the relevant context dynamically
        Problem 2 → TOOLS:      give the model access to live data sources
        Problem 3 → AGENTS:     let the model choose and execute tools

### What LangChain Actually Is

    LangChain is a composition framework. It does not implement any models —
    it wraps every major model provider (OpenAI, Anthropic, Google, Ollama,
    HuggingFace, etc.) with a unified interface and provides the glue code
    to compose them into applications.

    The key metaphor: LangChain is to LLMs what Express.js is to Node.js,
    or what SQLAlchemy is to databases. It abstracts the provider details
    and provides higher-level building blocks.

    Without LangChain, building a simple RAG chatbot requires:
        - Tokenisation and chunking logic
        - Embedding API calls
        - Vector database setup and querying
        - Prompt construction with retrieved context
        - LLM API calls with retry logic
        - Response parsing
        - Conversation history management
        - Error handling across all of the above

    LangChain provides pre-built, composable components for all of these.

### LangChain vs Direct API Calls

    Direct API approach:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Answer: {question}"}]
        )
        answer = response.choices[0].message.content

    LangChain approach (same result, but composable, swappable, observable):
        chain = prompt_template | llm | output_parser
        answer = chain.invoke({"question": question})

    The LangChain version is:
        - SWAPPABLE: change llm to another provider in one line
        - OBSERVABLE: LangSmith automatically logs every call
        - COMPOSABLE: chain can be extended with retrieval, memory, tools
        - TYPED: input/output schemas are validated
        - STREAMING: works with .stream() without code changes

### LangChain's Architecture Layers

    Layer 1 — langchain-core:
        The abstract contracts. Runnable protocol. BaseMessage types.
        No model-specific code. Minimal dependencies.

    Layer 2 — langchain (main library):
        Chains (LLMChain, SequentialChain, etc.)
        Agents and agent executors
        Memory implementations
        RetrievalQA and other RAG chains
        Text splitters and document transformers

    Layer 3 — langchain-community:
        600+ integrations: vector stores, document loaders, tools,
        LLM wrappers, embeddings, chat models.
        Third-party contributed, community-maintained.

    Layer 4 — Partner packages:
        langchain-openai, langchain-anthropic, langchain-google-genai, etc.
        Official, first-party integration packages.

    Layer 5 — LangGraph (separate package):
        Framework for stateful, cyclic agent graphs.
        The recommended way to build complex multi-step agents in 2024+.
        Think of it as LangChain for agentic workflows with explicit state.

    Layer 6 — LangSmith (SaaS observability platform):
        Tracing, evaluation, dataset management for LangChain applications.
        Every chain invocation is logged with inputs, outputs, latencies,
        token counts, and the full trace of every nested call.


##### PART 2 — THE RUNNABLE PROTOCOL AND LCEL

### What Is a Runnable?

    The Runnable protocol is the foundational abstraction in LangChain 0.2+.
    Every component — prompts, models, output parsers, retrievers, custom
    functions — implements the Runnable interface:

        .invoke(input)        → output               (single, synchronous)
        .batch(inputs)        → list[output]          (parallel, synchronous)
        .stream(input)        → Iterator[output]      (streaming)
        .ainvoke(input)       → Awaitable[output]     (async single)
        .abatch(inputs)       → Awaitable[list]       (async parallel)
        .astream(input)       → AsyncIterator         (async streaming)

    The power: because EVERYTHING is Runnable, EVERYTHING composes.

### LCEL: LangChain Expression Language

    LCEL is a domain-specific language expressed entirely through Python's
    pipe operator (|). It allows building chains declaratively:

        chain = prompt | model | output_parser

    The pipe operator (|) is Python's __or__ method. LangChain overloads it
    to mean "pass the output of the left as the input of the right."

    Reading a chain:
        prompt         → takes a dict, returns a PromptValue (formatted text)
        model          → takes a PromptValue, returns a BaseMessage (AIMessage)
        output_parser  → takes a BaseMessage, returns final parsed output

    The entire chain is itself a Runnable — it has .invoke(), .stream(), etc.
    This is the key composability property: chains of chains work identically.

### LCEL Composition Patterns

    1. Sequential pipe (the most common):
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"topic": "quantum computing"})

    2. Parallel execution (RunnableParallel):
        from langchain_core.runnables import RunnableParallel

        parallel_chain = RunnableParallel(
            summary   = summary_chain,
            sentiment = sentiment_chain,
            keywords  = keyword_chain,
        )
        result = parallel_chain.invoke({"text": article})
        # result = {"summary": "...", "sentiment": "...", "keywords": [...]}
        # All three chains run SIMULTANEOUSLY

    3. Passthrough (preserve original input):
        from langchain_core.runnables import RunnablePassthrough

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        # retriever gets the question, RunnablePassthrough passes it through
        # Both arrive at the prompt as {"context": docs, "question": query}

    4. Lambda / custom function:
        from langchain_core.runnables import RunnableLambda

        format_docs = RunnableLambda(lambda docs: "\n\n".join(d.page_content for d in docs))
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    5. Conditional branching (RunnableBranch):
        branch = RunnableBranch(
            (lambda x: "python" in x["query"].lower(),  python_chain),
            (lambda x: "sql"    in x["query"].lower(),  sql_chain),
            default_chain,     # fallback
        )

    6. Fallbacks (error resilience):
        robust_chain = (openai_chain
                        .with_fallbacks([anthropic_chain, local_chain]))

    7. Retry logic:
        with_retry = chain.with_retry(stop_after_attempt=3,
                                       wait_exponential_jitter=True)

### Input and Output Schemas

    Every Runnable has typed input and output schemas:
        chain.input_schema    → Pydantic model describing input
        chain.output_schema   → Pydantic model describing output
        chain.get_graph()     → networkx graph of the full chain DAG

    This enables automatic OpenAPI documentation, input validation,
    and LangServe's automatic /docs endpoint.

### Configurable Runnables

    Runnables can be made runtime-configurable without changing code:

        configurable_llm = llm.configurable_alternatives(
            ConfigurableField(id="model"),
            default_key = "openai",
            anthropic   = ChatAnthropic(model="claude-3-5-sonnet-20241022"),
            local       = Ollama(model="llama3.2"),
        )

        chain = prompt | configurable_llm | StrOutputParser()

        # At runtime, select different models:
        chain.invoke(input, config={"configurable": {"model": "anthropic"}})
        chain.invoke(input, config={"configurable": {"model": "local"}})


##### PART 3 — MODELS, MESSAGES, AND PROMPT TEMPLATES

### LLM vs ChatModel: The Two Model Types

    LangChain has two base model types:

    LLM (text completion models):
        Input: a string.  Output: a string.
        Legacy interface — corresponds to models like text-davinci-003.
        from langchain_openai import OpenAI
        llm = OpenAI(model="gpt-3.5-turbo-instruct")
        result = llm.invoke("What is the capital of France?")
        # returns: "Paris"

    ChatModel (conversational models — the modern standard):
        Input: a list of BaseMessage objects.  Output: an AIMessage.
        Corresponds to gpt-4, claude-3, llama-3, etc.
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model="gpt-4o-mini")
        result = model.invoke([HumanMessage(content="What is Paris?")])
        # returns: AIMessage(content="Paris is the capital of France...")

    Modern practice: always use ChatModel. The chat interface has superseded
    pure completion models across all major providers.

### The Message Type System

    LangChain has a typed message system matching the OpenAI chat format:

        SystemMessage:     sets context, persona, or instructions for the AI
        HumanMessage:      user's input (role="user")
        AIMessage:         model's response (role="assistant")
        FunctionMessage:   result from a function/tool call (legacy)
        ToolMessage:       result from a tool call (modern, preferred)

    Additional types:
        ChatMessage:       generic message with arbitrary role string
        BaseMessage:       abstract base for all message types

    Direct invocation:
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage("You are a concise, helpful assistant."),
            HumanMessage("Explain gradient descent in one sentence."),
        ]
        response = model.invoke(messages)   # AIMessage
        response.content                    # the text string
        response.usage_metadata             # token counts

### Prompt Templates: Reusable, Versioned Prompts

    PromptTemplate (for LLMs — string output):
        from langchain_core.prompts import PromptTemplate

        template = PromptTemplate.from_template(
            "Write a {length} summary of the following text:\n\n{text}"
        )
        prompt_value = template.invoke({"length": "two-sentence", "text": article})

    ChatPromptTemplate (for ChatModels — list of messages output):
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages([
            ("system", "You are a {persona} expert. Respond in {language}."),
            ("human",  "{question}"),
        ])
        # String tuples automatically converted to SystemMessage / HumanMessage

        messages = template.invoke({
            "persona": "machine learning",
            "language": "English",
            "question": "What is backpropagation?",
        })

    MessagesPlaceholder — inject a list of messages at a specific position:
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),    # ← inject history here
            ("human", "{question}"),
        ])

        # Inject previous conversation turns:
        messages = template.invoke({
            "history": [
                HumanMessage("What is the capital of France?"),
                AIMessage("Paris."),
            ],
            "question": "What is its population?"
        })

### Few-Shot Prompting

    FewShotChatMessagePromptTemplate:
        Dynamically selects the most relevant examples to include.
        Uses a semantic similarity search to pick examples closest to the input.

        from langchain_core.prompts import FewShotChatMessagePromptTemplate

        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "2+3", "output": "5"},
            {"input": "5*5", "output": "25"},
        ]
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"), ("ai", "{output}")
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples        = examples,
            example_prompt  = example_prompt,
        )
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a math tutor."),
            few_shot_prompt,
            ("human", "{question}"),
        ])

### Output Parsers: Structured Extraction

    Raw LLM output is text. Output parsers convert it to typed structures.

    StrOutputParser (most common):
        Extracts the .content string from an AIMessage.
        chain = prompt | llm | StrOutputParser()

    JsonOutputParser:
        Parses JSON from the model output. Adds format instructions to the prompt.
        parser = JsonOutputParser(pydantic_object=MyPydanticModel)

    PydanticOutputParser:
        Parses structured data into a Pydantic model instance.
        Adds schema instructions to the prompt automatically.

        from langchain_core.output_parsers import PydanticOutputParser
        from pydantic import BaseModel, Field

        class MovieReview(BaseModel):
            title:     str   = Field(description="Movie title")
            rating:    float = Field(description="Rating 1-10")
            sentiment: str   = Field(description="positive/negative/neutral")

        parser = PydanticOutputParser(pydantic_object=MovieReview)
        chain  = prompt | llm | parser
        review = chain.invoke({"text": "I loved Inception!"})
        # review: MovieReview(title="Inception", rating=9.5, sentiment="positive")

    Structured output (modern, native):
        # Preferred over PydanticOutputParser for models that support it
        structured_llm = llm.with_structured_output(MovieReview)
        review = structured_llm.invoke("Review: I loved Inception!")

    CommaSeparatedListOutputParser, XMLOutputParser, MarkdownListOutputParser
    are available for simpler extraction needs.


##### PART 4 — RETRIEVAL AUGMENTED GENERATION (RAG): ARCHITECTURE

### What Is RAG and Why It Exists

    Retrieval Augmented Generation (RAG) is the pattern of augmenting an LLM's
    response with dynamically retrieved context from an external knowledge base.

    The problem it solves:
        - Model trained on data until 2023 cannot answer about 2024 events
        - Model cannot know your internal company documents
        - Stuffing all possible context into the prompt is expensive and impossible
        - Fine-tuning a model on proprietary data is costly and produces hallucinations

    The RAG solution:
        1. INGEST: process your documents into a searchable index
        2. RETRIEVE: at query time, find the most relevant documents
        3. AUGMENT: inject retrieved docs into the LLM's context window
        4. GENERATE: LLM answers using the retrieved context
        5. CITE: optionally include source references

    RAG does NOT update the model's weights. It updates the model's context.
    The model reasons over retrieved facts, producing grounded responses.

### The RAG Pipeline in Detail

    INGESTION PHASE (done once, offline):

        1. Document Loading:
            Load from PDF, Word, web pages, databases, APIs.
            DocumentLoader → list[Document]
            Document = {page_content: str, metadata: dict}

        2. Text Splitting:
            Split long documents into overlapping chunks.
            Why: embedding models have max length limits (512–8192 tokens).
            Why: retrieval works better on focused chunks than giant documents.
            RecursiveCharacterTextSplitter → list[Document]

        3. Embedding:
            Convert each chunk into a dense vector representation.
            Semantically similar chunks → nearby vectors in embedding space.
            Embedding model: text-embedding-ada-002, sentence-transformers, etc.

        4. Vector Storage:
            Store chunk text + embedding vector in a vector database.
            Supports approximate nearest-neighbour search (ANN).
            FAISS, Chroma, Pinecone, Weaviate, Qdrant, pgvector, etc.

    RETRIEVAL PHASE (done at query time):

        5. Query Embedding:
            Embed the user's question into the same vector space.

        6. Similarity Search:
            Find the k chunks whose embedding vectors are most similar
            to the query vector (cosine similarity or L2 distance).
            Returns: top-k Document objects.

        7. Re-ranking (optional but recommended):
            Re-score the top-k candidates with a cross-encoder model.
            Cross-encoder is more accurate than bi-encoder but slower.
            Typical flow: retrieve top-50, re-rank to top-5.

    GENERATION PHASE:

        8. Context Formatting:
            Concatenate retrieved chunks with separators.
            Build prompt: "Given this context: {chunks}\nAnswer: {question}"

        9. LLM Generation:
            Model sees question + retrieved context.
            Produces grounded, citable answer.

        10. Source Attribution:
            Extract source metadata from retrieved documents.
            Return answer + list of sources.

### Text Splitting Strategies

    RecursiveCharacterTextSplitter (recommended for most cases):
        Tries to split on: ["\\n\\n", "\\n", " ", ""] in order.
        Preserves paragraph > sentence > word structure.
        Parameters:
            chunk_size:    target chunk size in characters (500–2000 typical)
            chunk_overlap: overlap between consecutive chunks (10–20% of chunk_size)
            length_function: len (characters) or token_counter (for model limits)

    Overlap is critical:
        If a relevant sentence spans a chunk boundary, overlap ensures at
        least one of the two chunks contains the full sentence.
        Rule of thumb: chunk_overlap = 10–20% of chunk_size.

    TokenTextSplitter:
        Split by actual token count — guarantees fit within model context.
        Slower (requires tokenisation) but more precise for token-limited embeds.

    MarkdownHeaderTextSplitter:
        Splits on Markdown headers (# ## ###).
        Preserves document structure. Headers added to metadata for context.
        Excellent for structured documentation.

    SemanticChunker:
        Splits where embedding similarity drops between consecutive sentences.
        Produces semantically coherent chunks rather than fixed-size windows.
        Requires an embedding model. More expensive but better retrieval quality.

### Retrieval Strategies

    Similarity search (baseline):
        Cosine similarity between query and stored embeddings.
        Fast but can miss lexically different but semantically similar chunks.

    MMR — Maximum Marginal Relevance:
        Balances relevance AND diversity in retrieved chunks.
        Iteratively selects the most relevant chunk that is also most different
        from already-selected chunks. Reduces redundancy.
        retriever = vectorstore.as_retriever(search_type="mmr", k=5)

    Multi-query retrieval:
        Generate 3-5 different phrasings of the question, retrieve for each.
        Union of results → higher recall.
        from langchain.retrievers import MultiQueryRetriever

    Contextual compression:
        After retrieval, compress each document to only the relevant portion.
        Reduces noise fed to the LLM.
        from langchain.retrievers import ContextualCompressionRetriever

    Hybrid search (BM25 + dense):
        Combine keyword-based (BM25/TF-IDF) with semantic (dense) retrieval.
        BM25 excels on exact keyword matches; dense retrieval on semantics.
        Reciprocal Rank Fusion combines the ranked lists.

    Self-query retrieval:
        LLM interprets structured filters from natural language queries.
        "Papers about neural networks from 2023" →
        search = {"year": 2023} + semantic="neural networks"

    Parent-document retrieval:
        Index small chunks for precision retrieval.
        But return the full parent document chunk to the LLM for more context.
        Retrieves precisely, generates with full context.

### Embedding Models and Vector Stores

    Embedding models:
        OpenAI: text-embedding-3-small (1536d, cheapest), text-embedding-3-large (3072d)
        Cohere: embed-english-v3.0 — competitive with OpenAI
        Sentence-transformers: all-MiniLM-L6-v2 (384d, free, fast, CPU)
                               all-mpnet-base-v2 (768d, free, higher quality)
                               BAAI/bge-large-en-v1.5 — near OpenAI quality, free
        Google: text-embedding-004 (768d)
        Voyage AI: voyage-3-lite — excellent quality for RAG

    Vector stores comparison:
        FAISS (Facebook AI Similarity Search):
            In-memory, no server required. Perfect for development and small datasets.
            Supports GPU indexing. NOT persistent (save/load manually).
            from langchain_community.vectorstores import FAISS

        ChromaDB:
            Embedded in Python process. Persistent with SQLite. No server needed.
            Best for prototyping and small-medium production use.
            from langchain_chroma import Chroma

        Pinecone:
            Managed cloud service. Serverless or pod-based. Production scale.
            Fully managed ANN index. Handles billions of vectors.
            from langchain_pinecone import PineconeVectorStore

        Qdrant:
            Open-source, can be self-hosted or cloud. Strong filtering.
            Supports named vectors (multiple embedding spaces per document).
            from langchain_qdrant import QdrantVectorStore

        pgvector:
            PostgreSQL extension. If you already run Postgres, add vectors to it.
            Transactional guarantees. Perfect for metadata-heavy filtering.
            from langchain_postgres import PGVector

        Weaviate:
            Open-source + cloud. Built-in BM25 hybrid search. GraphQL API.
            Good for complex filtering + semantic search combinations.


##### PART 5 — MEMORY: STATEFUL CONVERSATION

### The Statelessness Problem

    LLMs are stateless: each API call is independent. The model has no memory
    of previous turns unless you explicitly include them in the prompt.

    Without memory:
        Turn 1: "My name is Alice."  → "Hello Alice!"
        Turn 2: "What is my name?"   → "I don't know your name."

    With memory:
        Turn 1: "My name is Alice."  → "Hello Alice!"
        Turn 2: (history injected into prompt) → "Your name is Alice."

    The memory challenge: conversations grow without bound. Including the full
    history would eventually exceed the context window.

### Memory Strategies

    ConversationBufferMemory:
        Stores ALL messages verbatim.
        Simple. Gets large fast. Hits context limit for long conversations.
        Good for: short sessions, testing, development.

    ConversationBufferWindowMemory(k=5):
        Stores only the last k conversation turns.
        Fixed memory size. Forgets early context.
        Good for: multi-turn chat where recent context is sufficient.

    ConversationSummaryMemory:
        Summarises old turns using an LLM as context grows.
        Keeps a running summary + recent turns verbatim.
        Smart compression. Requires LLM calls for summarisation.
        Good for: long conversations where early context matters.

    ConversationSummaryBufferMemory:
        Hybrid: keep recent turns verbatim (fast) + summarise older ones.
        Best of both worlds. The recommended default for production.

    ConversationTokenBufferMemory:
        Keeps as many recent messages as fit within a token budget.
        More precise than window-based memory.

    VectorStoreRetrieverMemory:
        Store all past turns as embeddings. Retrieve the semantically
        most relevant memories when needed. Long-term associative memory.
        Good for: very long sessions, personalisation across sessions.

    Entity Memory:
        Extracts key entities (people, places, concepts) and maintains
        a profile for each. "Alice is a software engineer in Berlin" is
        stored and retrieved when Alice is mentioned later.

### Modern Memory Pattern (LangChain 0.2+)

    The modern pattern uses RunnableWithMessageHistory:

        from langchain_core.runnables.history import RunnableWithMessageHistory
        from langchain_community.chat_message_histories import ChatMessageHistory

        # session_id → message history store
        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key  = "input",
            history_messages_key = "history",
        )

        # Different session_ids = different conversation histories
        with_memory.invoke(
            {"input": "My name is Alice."},
            config={"configurable": {"session_id": "user_123"}}
        )
        with_memory.invoke(
            {"input": "What is my name?"},
            config={"configurable": {"session_id": "user_123"}}
        )
        # Returns: "Your name is Alice."


##### PART 6 — AGENTS: LLMs THAT ACT

### What Is an Agent?

    A chain has a fixed execution path: step 1 → step 2 → step 3.
    An agent has a dynamic execution path: the LLM decides what to do next.

    The agent loop (ReAct pattern — Reason + Act):

        while not done:
            1. THINK:   LLM observes current state and reasons about next step
            2. ACT:     LLM calls a tool (specifies tool name + arguments)
            3. OBSERVE: Tool executes and returns result
            4. REPEAT:  LLM sees tool result, decides whether to act again or answer

    This loop continues until the LLM decides it has enough information
    to produce the final answer (or a maximum iteration limit is reached).

    Example — "What is the population of the capital of France, as of today?":
        Thought: I need to find the capital of France, then look up its population.
        Action: Search("capital of France")
        Observation: "The capital of France is Paris."
        Thought: Now I need the current population of Paris.
        Action: Search("Paris population 2024")
        Observation: "Paris has approximately 2.1 million people in the city proper."
        Thought: I have everything I need.
        Answer: The capital of France is Paris, with a population of ~2.1M.

### Tools: Capabilities the Agent Can Use

    A Tool is a Python function with:
        name:        unique identifier (the LLM uses this to call the tool)
        description: what the tool does (the LLM reads this to decide when to use it)
        func:        the actual Python function to execute

    Defining tools:
        1. Using @tool decorator (recommended):

            from langchain_core.tools import tool

            @tool
            def get_weather(city: str) -> str:
                # Docstring IS the tool description — write it carefully!
                # "Get the current weather for a given city.
                # Returns temperature in Celsius and weather description."
                return weather_api.get(city)

        2. Using StructuredTool (for complex inputs):

            from langchain_core.tools import StructuredTool
            from pydantic import BaseModel

            class SearchInput(BaseModel):
                query: str
                max_results: int = 5

            search_tool = StructuredTool.from_function(
                func        = search_function,
                name        = "web_search",
                description = "Search the web for current information.",
                args_schema = SearchInput,
            )

    Built-in tools (from langchain_community.tools):
        DuckDuckGoSearchRun:    web search (no API key needed)
        WikipediaQueryRun:      Wikipedia article retrieval
        PythonREPLTool:         execute Python code in a sandbox
        ShellTool:              execute shell commands
        FileManagementToolkit:  read/write/list files
        HumanInputRun:          ask a human for clarification
        ArxivQueryRun:          search academic papers
        PubmedQueryRun:         search medical literature
        SQLDatabaseToolkit:     query SQL databases with natural language

### Agent Types

    ReAct Agent (the standard for tool use):
        Prompt engineering approach: model is prompted with the ReAct format.
        Works with any instruction-following LLM.
        Uses thought → action → observation loop explicitly in the prompt.

    OpenAI Functions / Tool Calling Agent:
        Uses OpenAI's native function calling / tool use feature.
        Model returns structured JSON for tool calls instead of free text.
        More reliable than prompt-based ReAct. Available on: GPT-4, Claude-3.
        This is the modern default for capable models.

    XML Agent:
        Uses XML tags for reasoning. Works with Claude models specifically.

    Conversational Agent:
        Maintains conversation history while also using tools.
        Remembers what was discussed, can clarify ambiguities.

### Tool Calling Models (Modern Standard)

    Modern LLMs (GPT-4, Claude-3, Llama-3.1, Mistral-Large) support
    native tool calling: the model returns a structured tool_call object
    rather than free text.

        tools = [get_weather, web_search, calculator]
        llm_with_tools = llm.bind_tools(tools)

        # Model decides if/which tool to call:
        response = llm_with_tools.invoke("What's 42 * 17?")
        # response.tool_calls = [{"name": "calculator", "args": {"expression": "42*17"}}]

    The tool calling agent:
        from langchain.agents import create_tool_calling_agent, AgentExecutor

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                  max_iterations=10, handle_parsing_errors=True)
        result = executor.invoke({"input": "What is the capital of France's population?"})

### LangGraph: The Modern Agent Framework

    For complex multi-step agents, LangGraph provides:
        - Explicit state graph: define agent state as a TypedDict
        - Nodes: functions that transform the state
        - Edges: conditions that determine which node runs next
        - Cycles: graphs can loop (unlike LCEL which is DAG-only)
        - Persistence: built-in checkpointing for long-running agents
        - Human-in-the-loop: pause execution for human approval

    LangGraph mental model:
        state = {"messages": [...], "context": "...", "step": 0}
        graph = StateGraph(State)
        graph.add_node("retrieve", retrieve_fn)
        graph.add_node("generate", generate_fn)
        graph.add_node("check_answer", check_fn)
        graph.add_conditional_edges("check_answer",
            lambda s: "done" if s["confident"] else "retrieve")
        app = graph.compile()
        result = app.invoke({"messages": [HumanMessage("...")]})


##### PART 7 — DOCUMENT LOADERS, CHAINS, AND EVALUATION

### Document Loaders: Ingesting Any Source

    LangChain has 200+ document loaders. They all return list[Document]:
        Document.page_content:  the text content
        Document.metadata:      source, page, author, date, etc.

    Common loaders:
        TextLoader:            plain text files
        PyPDFLoader:           PDF documents (via pypdf)
        PyMuPDFLoader:         faster PDF loader (via fitz)
        UnstructuredPDFLoader: handles complex layouts, tables, images
        Docx2txtLoader:        Word documents
        CSVLoader:             CSV files → one Document per row
        WebBaseLoader:         load HTML from URLs (uses beautifulsoup4)
        SeleniumURLLoader:     JavaScript-rendered pages
        WikipediaLoader:       Wikipedia articles
        YoutubeAudioLoader:    audio transcription via Whisper
        GitLoader:             entire Git repositories
        SlackDirectoryLoader:  Slack export archives
        NotionDirectoryLoader:  Notion exports
        ConfluenceLoader:      Atlassian Confluence spaces
        GoogleDriveLoader:     Google Drive files

    Pattern: load → split → embed → store (the indexing pipeline):

        from langchain_community.document_loaders import WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma

        loader   = WebBaseLoader("https://example.com/article")
        docs     = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks   = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vectordb   = Chroma.from_documents(chunks, embeddings)

### Legacy Chains (still used widely)

    LLMChain (the original workhorse):
        Combines a prompt template with an LLM.
        Equivalent to: prompt | llm | StrOutputParser()
        Still used in legacy code and documentation.

    SequentialChain:
        Passes output of one chain as input to next.
        Multiple chains in sequence.
        Equivalent to LCEL's pipe composition.

    RetrievalQA:
        The classic RAG chain. Retrieves docs + answers questions.
        Equivalent to LCEL's retriever + prompt + llm pattern.

    ConversationalRetrievalChain:
        RAG + conversation history.
        Rewrites the question with conversation context before retrieval.
        This question rewriting step is critical: "What about the second one?"
        must be rewritten to "What is the second recommendation mentioned?" for
        good retrieval.

    LLMMathChain:
        Uses Python REPL to solve mathematical equations via code generation.

    SQLDatabaseChain:
        Translates natural language to SQL, executes it, returns results.

### Evaluation in LangChain

    Evaluating LLM applications is hard: there is no ground truth for many tasks.
    LangChain provides several evaluation approaches:

    String evaluators (compare to reference):
        from langchain.evaluation import load_evaluator
        evaluator = load_evaluator("qa")
        result = evaluator.evaluate_strings(
            prediction="Paris is the capital.",
            reference="The capital of France is Paris.",
            input="What is the capital of France?"
        )

    Criteria evaluator (no reference needed):
        evaluator = load_evaluator("criteria", criteria="conciseness")
        # Evaluates if output is concise — uses an LLM as judge

    Embedding distance evaluator:
        Measures semantic similarity between prediction and reference.
        from langchain.evaluation import load_evaluator
        evaluator = load_evaluator("embedding_distance")

    LangSmith evaluation (the recommended production approach):
        - Define a dataset of input/output pairs
        - Run your chain on the dataset
        - Apply evaluators (LLM-as-judge, human review, regex, etc.)
        - Compare across different chain versions


##### PART 8 — PRODUCTION: LANGSMITH, LANGSERVE, AND BEST PRACTICES

### LangSmith: Observability for LLM Applications

    LLM applications fail in subtle ways: wrong retrieved context, model
    hallucinating despite good context, prompt injections, tool call failures.
    Standard logging is insufficient — you need to see the full trace.

    LangSmith automatically captures every invocation:
        - Full input and output at every step
        - Latency per step and total
        - Token counts and estimated cost
        - Error messages with full stack traces
        - Complete nested call graph (chain → prompt → LLM → parser)

    Setup:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"]    = "your-langsmith-key"
        os.environ["LANGCHAIN_PROJECT"]    = "my-rag-app"

        # Now every chain.invoke() is automatically traced — no code changes

    LangSmith datasets and evaluation:
        - Create datasets from traces (real user queries)
        - Run evaluation suites automatically
        - Compare model A vs model B on the same dataset
        - Track performance regressions across deployments

### LangServe: Deploy Chains as REST APIs

    LangServe wraps any Runnable chain into a FastAPI server
    with automatic OpenAPI docs and playground.

        from fastapi import FastAPI
        from langserve import add_routes

        app   = FastAPI()
        chain = prompt | llm | StrOutputParser()

        add_routes(app, chain, path="/chat")

        # Auto-generated endpoints:
        # POST /chat/invoke      → single invocation
        # POST /chat/batch       → parallel batch invocation
        # POST /chat/stream      → streaming SSE
        # GET  /chat/playground  → interactive browser UI
        # GET  /chat/input_schema  → JSON schema of input
        # GET  /chat/output_schema → JSON schema of output

    Deploy:
        uvicorn my_app:app --host 0.0.0.0 --port 8080

    LangServe Remote Runnable (client):
        from langserve import RemoteRunnable
        remote_chain = RemoteRunnable("http://localhost:8080/chat")
        result = remote_chain.invoke({"question": "Hello?"})
        # Works identically to local chain — same .invoke()/.stream()/.batch()

### Production Best Practices

    1. ALWAYS use streaming for user-facing responses:
        async for chunk in chain.astream({"question": query}):
            print(chunk, end="", flush=True)
        # Users see tokens appear immediately — dramatically better UX.

    2. Add fallbacks for reliability:
        primary   = ChatOpenAI(model="gpt-4o")
        fallback  = ChatAnthropic(model="claude-3-5-sonnet-20241022")
        robust_llm = primary.with_fallbacks([fallback])

    3. Cache repeated calls:
        from langchain_community.cache import InMemoryCache
        from langchain_core.globals import set_llm_cache
        set_llm_cache(InMemoryCache())
        # Identical prompts return cached responses — saves cost and latency.
        # Use SQLiteCache for persistence across restarts.

    4. Implement retry logic:
        chain_with_retry = chain.with_retry(
            stop_after_attempt       = 3,
            wait_exponential_jitter  = True,
            retry_if_exception_type  = (openai.RateLimitError,)
        )

    5. Rate limiting and budget control:
        callbacks = [OpenAICallbackHandler()]
        result = chain.invoke(input, config={"callbacks": callbacks})
        print(f"Total tokens: {callbacks[0].total_tokens}")
        print(f"Total cost: ${callbacks[0].total_cost:.4f}")

    6. Structured output > output parsers for reliable extraction:
        # Using native tool calling is more reliable than regex/JSON parsing
        structured_llm = llm.with_structured_output(MySchema, method="json_mode")

    7. Tune chunk size for your embedding model:
        # OpenAI ada-002: up to 8191 tokens. Optimal chunk: 512–1000 tokens.
        # all-MiniLM-L6-v2: max 256 tokens. Must use smaller chunks.

    8. Re-rank after retrieval:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_cohere import CohereRerank
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    9. Always add max_iterations to agents:
        executor = AgentExecutor(agent=agent, tools=tools,
                                  max_iterations=10,   # prevents infinite loops
                                  max_execution_time=30)  # seconds

    10. Use async everywhere for production throughput:
        # Sync: 1 request processed at a time
        # Async: 100s of requests concurrently
        results = await chain.abatch(list_of_inputs, config={"max_concurrency": 10})

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · LCEL & Chains — Runnable Protocol, Prompts & Output Parsers": {
        "description": (
            "Complete LangChain Expression Language (LCEL) tour. "
            "Runnable protocol: .invoke(), .batch(), .stream() interface. "
            "PromptTemplate and ChatPromptTemplate construction. "
            "Pipe composition: prompt | llm | parser end-to-end. "
            "RunnableParallel: simultaneous multi-chain execution. "
            "RunnablePassthrough: preserving input through the pipeline. "
            "RunnableLambda: wrapping Python functions as Runnables. "
            "RunnableBranch: conditional routing by input content. "
            "PydanticOutputParser and with_structured_output. "
            "Streaming with .stream() token by token. "
            "Fallbacks and retry logic for production robustness. "
            "Simulated LLM for offline demonstration."
        ),
        "language": "python",
        "code": '''
import time
import json
from typing import Iterator, Any

try:
    from langchain_core.prompts import (
        ChatPromptTemplate, PromptTemplate,
        FewShotChatMessagePromptTemplate, MessagesPlaceholder,
    )
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import (
        RunnablePassthrough, RunnableLambda, RunnableParallel,
        RunnableBranch,
    )
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import BaseModel, Field
    print("  langchain-core: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'langchain', 'langchain-core', '--quiet'], check=True)
    from langchain_core.prompts import (
        ChatPromptTemplate, PromptTemplate, MessagesPlaceholder,
        FewShotChatMessagePromptTemplate,
    )
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.runnables import (
        RunnablePassthrough, RunnableLambda, RunnableParallel,
        RunnableBranch,
    )
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import BaseModel, Field

print("=" * 65)
print("  LCEL & CHAINS — RUNNABLE PROTOCOL, PROMPTS & PARSERS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# MOCK LLM — fully functional simulation for offline demonstration
# ─────────────────────────────────────────────────────────────────────────

class MockChatModel(BaseChatModel):
    """
    A fully functional mock LLM that simulates deterministic responses.
    Demonstrates the full LangChain interface without requiring API keys.
    """
    model_name: str = "mock-gpt-4"
    responses: dict = Field(default_factory=dict)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Examine the last human message to produce relevant response
        last_msg = ""
        for m in reversed(messages):
            if hasattr(m, "content") and m.content:
                last_msg = m.content.lower()
                break

        content = self._pick_response(last_msg, messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _pick_response(self, text, messages):
        if "summarize" in text or "summary" in text:
            return "This document covers key concepts in machine learning, including supervised learning, neural networks, and gradient descent. The main takeaway is that models learn from data through iterative optimisation."
        if "sentiment" in text:
            return "positive"
        if "keywords" in text or "keyword" in text:
            return "machine learning, neural networks, gradient descent, optimisation"
        if "translate" in text and "french" in text:
            return "L'apprentissage automatique est une branche de l'intelligence artificielle."
        if "python" in text or "code" in text:
            return "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)"
        if "sql" in text:
            return "SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY name;"
        if "json" in text or "structure" in text or "extract" in text:
            return json.dumps({"title": "Inception", "rating": 9.2,
                               "sentiment": "positive", "genre": "sci-fi"})
        if "capital" in text and "france" in text:
            return "The capital of France is Paris."
        if "weather" in text:
            return "The weather in Paris is currently 18°C and partly cloudy."
        if "hello" in text or "hi" in text:
            return "Hello! I am a mock language model. How can I help you?"
        return "I understand your query. Based on my analysis, the answer involves careful consideration of multiple factors. The key insight is that context matters enormously in these situations."

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        content = self._generate(messages).generations[0].message.content
        for word in content.split(" "):
            from langchain_core.outputs import ChatGenerationChunk
            from langchain_core.messages import AIMessageChunk
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            time.sleep(0.02)

    @property
    def _llm_type(self): return "mock"
    @property
    def _identifying_params(self): return {"model_name": self.model_name}

llm = MockChatModel()
print(f"  MockChatModel initialised (simulates GPT-4 responses offline)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: The Runnable Protocol
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — The Runnable protocol: invoke, batch, stream")
print("━" * 65)
print()

# Every LangChain component implements .invoke()
prompt  = ChatPromptTemplate.from_template("Say hello to {name} in one sentence.")
parser  = StrOutputParser()

# Each component is independently invokable
prompt_out = prompt.invoke({"name": "Alice"})
print(f"  prompt.invoke() → {type(prompt_out).__name__}")
print(f"    messages[0]: {prompt_out.messages[0].content[:60]}")
print()

llm_out = llm.invoke(prompt_out)
print(f"  llm.invoke()   → {type(llm_out).__name__}")
print(f"    content: {llm_out.content[:60]}")
print()

parsed_out = parser.invoke(llm_out)
print(f"  parser.invoke() → {type(parsed_out).__name__}")
print(f"    result: {parsed_out[:60]}")
print()

# The LCEL pipe chains them automatically
chain = prompt | llm | parser
result = chain.invoke({"name": "Alice"})
print(f"  chain (prompt | llm | parser).invoke() → '{result[:60]}'")
print()

# .batch() — parallel processing of multiple inputs
batch_results = chain.batch([
    {"name": "Alice"},
    {"name": "Bob"},
    {"name": "the LangChain framework"},
])
print(f"  chain.batch() — {len(batch_results)} results:")
for name_input, res in zip(["Alice", "Bob", "LangChain"], batch_results):
    print(f"    {name_input:<15}: {res[:50]}")
print()

# .stream() — token-by-token streaming
print("  chain.stream() — token-by-token (simulated):")
print("    ", end="")
for chunk in chain.stream({"name": "the world"}):
    print(chunk, end="", flush=True)
print()
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: ChatPromptTemplate patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — ChatPromptTemplate: system, human, MessagesPlaceholder")
print("━" * 65)
print()

# Full chat template with system + history + human
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {persona}. Always respond in {language}."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

messages_out = chat_template.invoke({
    "persona":  "Python tutor",
    "language": "English",
    "history": [
        HumanMessage(content="What is a list in Python?"),
        AIMessage(content="A list is an ordered, mutable collection of items."),
    ],
    "question": "How do I append to a list?",
})

print(f"  ChatPromptTemplate with system + history + human:")
print(f"  {'─'*55}")
for msg in messages_out.messages:
    role    = type(msg).__name__.replace("Message", "")
    content = msg.content[:60] + ("..." if len(msg.content) > 60 else "")
    print(f"  [{role:<8}] {content}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: RunnableParallel — simultaneous execution
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — RunnableParallel: simultaneous multi-chain execution")
print("━" * 65)
print()

article = """
Machine learning is a subset of artificial intelligence that enables computers
to learn from experience without being explicitly programmed. The field was
formally founded in 1956 and has grown enormously since the deep learning
revolution of 2012.
"""

summary_prompt  = ChatPromptTemplate.from_template(
    "Summarize this text in one sentence:\n{text}")
sentiment_prompt = ChatPromptTemplate.from_template(
    "What is the sentiment of this text? Reply with one word:\n{text}")
keyword_prompt  = ChatPromptTemplate.from_template(
    "Extract 3 keywords from this text, comma-separated:\n{text}")

# Build three independent chains
summary_chain   = summary_prompt   | llm | StrOutputParser()
sentiment_chain = sentiment_prompt | llm | StrOutputParser()
keyword_chain   = keyword_prompt   | llm | StrOutputParser()

# Execute all three in parallel
parallel = RunnableParallel(
    summary   = summary_chain,
    sentiment = sentiment_chain,
    keywords  = keyword_chain,
)

t0      = time.perf_counter()
results = parallel.invoke({"text": article})
t_ms    = (time.perf_counter() - t0) * 1000

print(f"  RunnableParallel executed 3 chains in {t_ms:.0f}ms:")
for key, val in results.items():
    print(f"    {key:<12}: {val.strip()[:65]}")
print()
print(f"  → All three chains ran concurrently (IO overlap)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: RunnablePassthrough and RunnableLambda
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — RunnablePassthrough and RunnableLambda")
print("━" * 65)
print()

# RunnablePassthrough: passes input unchanged to the next component
translate_prompt = ChatPromptTemplate.from_template(
    "Translate to French: {text}")

passthrough_chain = (
    RunnableParallel(
        text         = RunnablePassthrough(),          # pass original text
        translation  = translate_prompt | llm | StrOutputParser(),  # also translate
    )
)

pt_result = passthrough_chain.invoke(
    "Machine learning is a branch of artificial intelligence."
)
print(f"  RunnablePassthrough example:")
print(f"    Original:    {pt_result['text'][:60]}")
print(f"    Translation: {pt_result['translation'][:60]}")
print()

# RunnableLambda: wrap any Python function as a Runnable
def count_words(text: str) -> dict:
    words  = text.split()
    unique = set(w.lower().strip(".,!?") for w in words)
    return {"text": text, "word_count": len(words), "unique_words": len(unique)}

def format_analysis(data: dict) -> str:
    return (f"Text has {data['word_count']} words "
            f"({data['unique_words']} unique). Content: {data['text'][:40]}...")

analysis_chain = (
    RunnableLambda(count_words)
    | RunnableLambda(format_analysis)
)

lambda_result = analysis_chain.invoke(
    "The quick brown fox jumps over the lazy dog the fox is quick"
)
print(f"  RunnableLambda pipeline:")
print(f"    Result: {lambda_result}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: RunnableBranch — conditional routing
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — RunnableBranch: conditional routing")
print("━" * 65)
print()

python_prompt = ChatPromptTemplate.from_template(
    "You are a Python expert. Answer this Python question: {question}")
sql_prompt    = ChatPromptTemplate.from_template(
    "You are a SQL expert. Answer this SQL question: {question}")
general_prompt = ChatPromptTemplate.from_template(
    "Answer this question: {question}")

python_chain  = python_prompt  | llm | StrOutputParser()
sql_chain     = sql_prompt     | llm | StrOutputParser()
general_chain = general_prompt | llm | StrOutputParser()

# Route based on the content of the input
router = RunnableBranch(
    (lambda x: "python" in x["question"].lower() or "def " in x["question"].lower(),
     python_chain),
    (lambda x: "sql"    in x["question"].lower() or "select" in x["question"].lower(),
     sql_chain),
    general_chain,   # default fallback
)

questions = [
    {"question": "How do I write a Python function that checks if a number is prime?"},
    {"question": "How do I write a SQL query to get recent users?"},
    {"question": "What is the capital of France?"},
]

print(f"  RunnableBranch routing by question type:")
routes_used = ["Python branch", "SQL branch", "General branch"]
for q, route in zip(questions, routes_used):
    result = router.invoke(q)
    print(f"  [{route:<15}] Q: '{q['question'][:40]}...'")
    print(f"  {' ':<17}  A: '{result.strip()[:55]}'")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Structured Output with Pydantic
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Structured output: PydanticOutputParser")
print("━" * 65)
print()

class MovieReview(BaseModel):
    title:     str   = Field(description="The title of the movie")
    rating:    float = Field(description="Rating from 0.0 to 10.0")
    sentiment: str   = Field(description="One of: positive, negative, neutral")
    genre:     str   = Field(description="Primary genre of the movie")

json_parser = JsonOutputParser(pydantic_object=MovieReview)

struct_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract structured information from movie reviews. "
               "Return valid JSON matching this schema: {format_instructions}"),
    ("human", "{review}"),
])

struct_chain = struct_prompt | llm | json_parser

review_texts = [
    "I absolutely loved Inception! It's a mind-bending sci-fi masterpiece.",
    "The Notebook was a touching romance, though a bit slow in parts.",
]

print(f"  PydanticOutputParser with MovieReview schema:")
for review_text in review_texts:
    result = struct_chain.invoke({
        "review": review_text,
        "format_instructions": json_parser.get_format_instructions(),
    })
    print(f"  Review: '{review_text[:50]}...'")
    print(f"  Parsed: {result}")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 7: Chain introspection and schema
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 7 — Chain introspection: schema and graph")
print("━" * 65)
print()

full_chain = chat_template | llm | StrOutputParser()

print(f"  Chain type:    {type(full_chain).__name__}")
print(f"  Input schema:  {full_chain.input_schema.schema()['properties'].keys()}")
print(f"  Output type:   string (StrOutputParser)")
print()

# Show the runnable graph steps
graph = full_chain.get_graph()
print(f"  Chain computation graph ({len(graph.nodes)} nodes):")
for i, node in enumerate(graph.nodes.values()):
    print(f"    [{i}] {node.name}")
print()

# Fallback chains for resilience
primary_llm  = MockChatModel(model_name="primary-gpt4")
fallback_llm = MockChatModel(model_name="fallback-claude")
robust_chain = (
    ChatPromptTemplate.from_template("{input}") | primary_llm | StrOutputParser()
).with_fallbacks([
    ChatPromptTemplate.from_template("{input}") | fallback_llm | StrOutputParser()
])
print(f"  Robust chain with fallback: {type(robust_chain).__name__}")
fallback_result = robust_chain.invoke({"input": "Hello world"})
print(f"  Fallback chain result: '{fallback_result[:40]}'")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · RAG Pipeline — Ingestion, Retrieval, and Generation": {
        "description": (
            "Complete end-to-end Retrieval Augmented Generation pipeline. "
            "Document loading with TextLoader and WebBaseLoader pattern. "
            "RecursiveCharacterTextSplitter with chunk_size and overlap. "
            "SemanticChunker concept and comparison. "
            "FAISS vector store creation from documents. "
            "Mock embedding model for offline demonstration. "
            "Similarity search vs MMR retrieval comparison. "
            "Full LCEL RAG chain: retriever | format | prompt | llm | parser. "
            "Conversational RAG with question rewriting for history. "
            "Parent document retrieval concept. "
            "Retrieval quality analysis: coverage and diversity metrics."
        ),
        "language": "python",
        "code": '''
import time
import re
import math
import json
from typing import List

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.embeddings import Embeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    print("  langchain + langchain-community: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'langchain', 'langchain-community',
                    'faiss-cpu', '--quiet'], check=True)
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.documents import Document
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.embeddings import Embeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

import sys, importlib
# Re-import mock LLM from previous operation context, or redefine
import numpy as np

print("=" * 65)
print("  RAG PIPELINE — INGESTION, RETRIEVAL, AND GENERATION")
print("=" * 65)
print()

# ── Mock LLM (same as Operation 1) ───────────────────────────────────
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage as AI
from pydantic import Field

class MockRAGLLM(BaseChatModel):
    model_name: str = "mock-rag-llm"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt_text = " ".join(
            m.content for m in messages if hasattr(m, "content")
        ).lower()
        if "context" in prompt_text and "transformer" in prompt_text:
            content = ("Based on the provided context, the Transformer architecture "
                       "uses self-attention mechanisms to process sequences in parallel, "
                       "replacing recurrent networks. It was introduced by Vaswani et al. "
                       "in the 2017 paper 'Attention Is All You Need'.")
        elif "context" in prompt_text and "bert" in prompt_text:
            content = ("According to the context, BERT (Bidirectional Encoder "
                       "Representations from Transformers) was released by Google in 2018. "
                       "It uses a masked language modelling objective for pretraining.")
        elif "rewrite" in prompt_text or "standalone" in prompt_text:
            content = "What are the key innovations of the Transformer architecture?"
        else:
            content = ("Based on the retrieved context, I can provide the following answer. "
                       "The documents indicate that this topic involves several important "
                       "considerations that are well-covered in the provided sources.")
        return ChatResult(generations=[ChatGeneration(message=AI(content=content))])

    @property
    def _llm_type(self): return "mock-rag"
    @property
    def _identifying_params(self): return {"model_name": self.model_name}

llm = MockRAGLLM()

# ── Mock Embeddings (deterministic TF-IDF style) ─────────────────────
class MockEmbeddings(Embeddings):
    """
    Deterministic mock embeddings based on word frequency.
    Produces consistent vectors so similarity search works meaningfully.
    """
    dim: int = 64

    def _text_to_vec(self, text: str) -> List[float]:
        import hashlib
        words = re.findall(r'\w+', text.lower())
        vec   = [0.0] * self.dim
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(3):
                idx = (h >> (i * 8)) % self.dim
                vec[idx] += 1.0
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._text_to_vec(t) for t in texts]

    def embed_query(self, text):
        return self._text_to_vec(text)

embeddings = MockEmbeddings()
print(f"  MockEmbeddings: {embeddings.dim}-dim deterministic vectors")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Document creation and knowledge base
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Document corpus: loading and metadata")
print("━" * 65)
print()

# Simulate a small technical knowledge base
RAW_DOCS = [
    {
        "content": """The Transformer architecture was introduced in the seminal 2017 paper
'Attention Is All You Need' by Vaswani et al. at Google Brain. The key innovation
was replacing recurrent layers entirely with self-attention mechanisms, which
allowed the model to process entire sequences in parallel rather than step-by-step.
This parallelism enabled training on much larger datasets and produced significantly
better results on machine translation benchmarks. The original Transformer used an
encoder-decoder structure: the encoder processes the input sequence bidirectionally,
while the decoder generates the output autoregressively with causal attention.""",
        "source": "transformers_intro.txt", "topic": "architecture"
    },
    {
        "content": """BERT (Bidirectional Encoder Representations from Transformers) was
released by Google in 2018. Unlike GPT models that use only the decoder,
BERT uses only the encoder portion of the Transformer. It is pretrained with
two objectives: Masked Language Modelling (MLM), where 15% of input tokens are
randomly masked and the model must predict them, and Next Sentence Prediction (NSP).
BERT representations are bidirectional: each token can attend to all other tokens
in both directions, making it excellent for understanding tasks like classification,
named entity recognition, and question answering.""",
        "source": "bert_overview.txt", "topic": "pretraining"
    },
    {
        "content": """GPT (Generative Pre-trained Transformer) models use only the
decoder portion of the Transformer architecture with causal (autoregressive)
attention. GPT-2 was released by OpenAI in 2019 with 1.5 billion parameters.
GPT-3 followed in 2020 with 175 billion parameters, demonstrating emergent
few-shot learning capabilities. GPT-4 in 2023 introduced multimodal capabilities.
These models are trained on next-token prediction: given all previous tokens,
predict the next one. This simple objective, when scaled massively, produces
models that can write code, answer questions, and reason across many domains.""",
        "source": "gpt_family.txt", "topic": "generation"
    },
    {
        "content": """Attention mechanisms compute a weighted sum of value vectors,
where the weights are determined by the compatibility between query and key vectors.
The formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.
Multi-head attention applies multiple attention functions in parallel, each
with different learned projections, then concatenates the results. This allows
the model to attend to information from different representation subspaces.
The scaling factor sqrt(d_k) prevents the dot products from growing too large
in magnitude, which would push the softmax into regions with very small gradients.""",
        "source": "attention_math.txt", "topic": "mathematics"
    },
    {
        "content": """Fine-tuning adapts a pretrained model to a specific task using
a much smaller labelled dataset. For BERT-style models, fine-tuning typically
involves adding a task-specific head (e.g., a classification layer) on top of
the pretrained representations and training end-to-end with a low learning rate
(typically 1e-5 to 5e-5) for 3-5 epochs. Parameter-Efficient Fine-Tuning (PEFT)
methods like LoRA reduce the computational cost by training only a small fraction
of parameters (often less than 1%) while achieving performance comparable to full
fine-tuning. This makes adapting large models much more accessible.""",
        "source": "finetuning_guide.txt", "topic": "training"
    },
    {
        "content": """Positional encoding adds location information to token embeddings
since self-attention is permutation-invariant. The original Transformer used
sinusoidal encoding: PE(pos, 2i) = sin(pos/10000^(2i/d_model)). Modern LLMs
favour Rotary Position Embedding (RoPE), which encodes position by rotating
query and key vectors in 2D subspaces. RoPE has become the dominant scheme
because it naturally encodes relative positions and enables length extrapolation
beyond the training context window. ALiBi adds position biases directly to
attention scores, requiring no learned position parameters at all.""",
        "source": "positional_encoding.txt", "topic": "architecture"
    },
]

# Create Document objects with metadata
documents = [
    Document(page_content=d["content"],
             metadata={"source": d["source"], "topic": d["topic"]})
    for d in RAW_DOCS
]

print(f"  Knowledge base: {len(documents)} documents")
for doc in documents:
    print(f"    {doc.metadata['source']:<30} topic={doc.metadata['topic']:<14}"
          f"chars={len(doc.page_content)}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Text splitting
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Text splitting: chunks, overlap, and metadata")
print("━" * 65)
print()

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 400,   # chars per chunk (use token splitter for production)
    chunk_overlap = 80,    # overlap between consecutive chunks
    length_function = len,
    separators    = ["\n\n", "\n", ". ", " ", ""],
)

chunks = splitter.split_documents(documents)

print(f"  RecursiveCharacterTextSplitter:")
print(f"    chunk_size={400}, chunk_overlap={80}")
print(f"    {len(documents)} documents → {len(chunks)} chunks")
print()
print(f"  Chunk length distribution:")
lengths = [len(c.page_content) for c in chunks]
print(f"    min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
print()
print(f"  First 3 chunks:")
for i, chunk in enumerate(chunks[:3]):
    print(f"  [{i}] source={chunk.metadata['source']}, len={len(chunk.page_content)}")
    print(f"      '{chunk.page_content[:80].strip()}...'")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Vector store creation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — FAISS vector store: indexing and similarity search")
print("━" * 65)
print()

t0       = time.perf_counter()
vectordb = FAISS.from_documents(chunks, embeddings)
t_index  = (time.perf_counter() - t0) * 1000

print(f"  FAISS index built from {len(chunks)} chunks in {t_index:.0f}ms")
print(f"  Index size: {vectordb.index.ntotal} vectors of dim {embeddings.dim}")
print()

# Similarity search
query          = "How does the Transformer use attention mechanisms?"
search_results = vectordb.similarity_search(query, k=3)

print(f"  Similarity search: '{query}'")
print(f"  Top 3 results:")
for i, doc in enumerate(search_results):
    print(f"  [{i}] source={doc.metadata['source']}")
    print(f"      '{doc.page_content[:100].strip()}...'")
    print()

# Similarity search with scores
results_with_scores = vectordb.similarity_search_with_score(query, k=4)
print(f"  Similarity scores (lower = more similar in FAISS L2):")
print(f"  {'Source':<30} {'Score':>8}")
print(f"  {'─'*40}")
for doc, score in results_with_scores:
    print(f"  {doc.metadata['source']:<30} {score:>8.4f}")
print()

# MMR — diversity-aware retrieval
mmr_retriever = vectordb.as_retriever(
    search_type   = "mmr",
    search_kwargs = {"k": 3, "fetch_k": 6, "lambda_mult": 0.5}
)
mmr_results = mmr_retriever.invoke(query)
print(f"  MMR retrieval (lambda=0.5, balances relevance + diversity):")
for doc in mmr_results:
    print(f"    topic={doc.metadata['topic']:<14} '{doc.page_content[:60].strip()}...'")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Full RAG chain with LCEL
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Full LCEL RAG chain")
print("━" * 65)
print()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    formatted = []
    for i, doc in enumerate(docs):
        formatted.append(
            f"[Source {i+1}: {doc.metadata['source']}]\n{doc.page_content}"
        )
    return "\n\n" + "\n\n---\n\n".join(formatted) + "\n"

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions based on the "
     "provided context. Always cite the source documents you used.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

# Full RAG chain
rag_chain = (
    {
        "context":  retriever | RunnableLambda(format_docs),  # retrieve + format
        "question": RunnablePassthrough(),                      # pass question through
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

test_questions = [
    "What is the Transformer architecture and who created it?",
    "How does BERT differ from GPT models?",
    "What is the mathematical formula for attention?",
]

print(f"  RAG chain: retriever | format | prompt | llm | parser")
print()
for question in test_questions:
    t0     = time.perf_counter()
    answer = rag_chain.invoke(question)
    t_ms   = (time.perf_counter() - t0) * 1000
    print(f"  Q: {question}")
    print(f"  A: {answer.strip()[:120]}...")
    print(f"  Latency: {t_ms:.0f}ms")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Conversational RAG with question rewriting
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Conversational RAG: question rewriting for history")
print("━" * 65)
print()

# Question rewriting: given chat history, rewrite the standalone question
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and a follow-up question, rewrite the follow-up "
     "question to be a standalone question that can be understood without the "
     "chat history. Return ONLY the rewritten question, nothing else."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Follow-up question: {question}"),
])

contextualize_chain = rewrite_prompt | llm | StrOutputParser()

def contextualize_question(input_dict):
    """Rewrite question if there is chat history, otherwise return as-is."""
    if input_dict.get("chat_history"):
        return contextualize_chain.invoke(input_dict)
    return input_dict["question"]

conversational_rag = (
    {
        "context":  RunnableLambda(contextualize_question) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"]),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Simulate a multi-turn conversation
chat_history = []

conversations = [
    "What is the Transformer architecture?",
    "How does its attention mechanism work mathematically?",  # refers to "its" = Transformer
    "What was the training objective used to pretrain BERT?",
]

print(f"  Conversational RAG (question rewriting preserves context):")
print()
for turn, question in enumerate(conversations):
    result = conversational_rag.invoke({
        "question":     question,
        "chat_history": chat_history,
    })
    print(f"  Turn {turn+1} — Q: '{question}'")
    if chat_history:
        rewritten = contextualize_chain.invoke({
            "question": question, "chat_history": chat_history
        })
        print(f"           Rewritten: '{rewritten.strip()[:70]}'")
    print(f"           A: '{result.strip()[:80]}...'")
    print()
    # Update chat history
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result),
    ])

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Retrieval quality metrics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Retrieval quality analysis")
print("━" * 65)
print()

eval_queries = {
    "transformer architecture": ["transformers_intro.txt", "attention_math.txt"],
    "BERT pretraining": ["bert_overview.txt"],
    "positional encoding": ["positional_encoding.txt", "transformers_intro.txt"],
    "LoRA fine-tuning": ["finetuning_guide.txt"],
}

print(f"  Retrieval coverage (k=3):")
print(f"  {'Query':<30} {'Expected sources':>20} {'Retrieved':>6} {'Hit':>6}")
print(f"  {'─'*65}")

total_queries, hits = 0, 0
for query_text, expected_sources in eval_queries.items():
    retrieved     = vectordb.similarity_search(query_text, k=3)
    retrieved_srcs = {d.metadata['source'] for d in retrieved}
    hit = any(src in retrieved_srcs for src in expected_sources)
    hits         += int(hit)
    total_queries += 1
    symbol = "✓" if hit else "✗"
    print(f"  [{symbol}] {query_text:<28} {str(expected_sources[0]):<22} "
          f"{len(retrieved):>6} {symbol:>6}")

recall = hits / total_queries
print(f"  {'─'*65}")
print(f"  Recall@3: {hits}/{total_queries} = {recall:.0%}  "
      f"(fraction of queries where expected source was retrieved)")
print()
print(f"  CHUNKING TUNING GUIDE:")
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │ chunk_size │ chunk_overlap │ Best for                         │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │ 200–400    │ 50–100        │ Dense technical docs, high recall  │")
print(f"  │ 500–1000   │ 100–200       │ General purpose (recommended)      │")
print(f"  │ 1000–2000  │ 200–400       │ Narrative text, needs more context │")
print(f"  │ 50–150     │ 20–50         │ Small embed models (MiniLM max 256)│")
print(f"  └─────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Memory & Agents — Conversation History and Tool-Using Agents": {
        "description": (
            "Memory systems for stateful conversation management. "
            "ConversationBufferMemory: full history verbatim. "
            "ConversationBufferWindowMemory(k=5): sliding window. "
            "ConversationSummaryMemory: LLM compresses old turns. "
            "RunnableWithMessageHistory: modern LCEL memory pattern. "
            "Session-based memory routing by session_id. "
            "Custom @tool functions with typed arguments. "
            "Tool creation patterns: @tool decorator vs StructuredTool. "
            "AgentExecutor with tool calling model. "
            "ReAct agent trace: Thought → Action → Observation loop. "
            "Multi-step agent solving a composite question. "
            "max_iterations and handle_parsing_errors for robustness."
        ),
        "language": "python",
        "code": '''
import time
import math
import json
from typing import Optional

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )
    print("  langchain + langchain-community: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'langchain', 'langchain-community', '--quiet'], check=True)
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    from langchain_core.tools import tool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )

print("=" * 65)
print("  MEMORY & AGENTS — CONVERSATION HISTORY AND TOOL USE")
print("=" * 65)
print()

# ── Mock LLM with memory and tool-call awareness ──────────────────────
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

class MockMemoryLLM(BaseChatModel):
    model_name: str = "mock-memory-llm"
    call_count: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        full_text = " ".join(
            m.content for m in messages if hasattr(m, "content") and m.content
        ).lower()

        # Count how many prior turns are visible
        human_turns  = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_turns     = sum(1 for m in messages if isinstance(m, AIMessage))

        if "name" in full_text and "alice" in full_text and "what is my name" in full_text:
            content = "Your name is Alice, as you mentioned earlier."
        elif "what is my name" in full_text:
            content = ("I don't have access to your name. "
                       "Could you tell me? [history_turns={} human, {} ai]"
                       ).format(human_turns, ai_turns)
        elif "summarize" in full_text or "summary" in full_text:
            content = ("The conversation so far has covered introductions and "
                       f"background information ({human_turns} turns recorded).")
        elif "calculate" in full_text or "math" in full_text:
            content = "I'll calculate that for you using the calculator tool."
        elif "weather" in full_text:
            content = "Let me check the weather for you."
        else:
            content = (f"Understood. [Turn {self.call_count}, "
                       f"history: {human_turns} human, {ai_turns} AI]")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self): return "mock-memory"
    @property
    def _identifying_params(self): return {}

llm = MockMemoryLLM()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: ConversationBufferMemory
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — ConversationBufferMemory: full verbatim history")
print("━" * 65)
print()

buffer_memory = ConversationBufferMemory(return_messages=True,
                                          memory_key="history")

# Manually add turns to show memory accumulation
convs = [
    ("My name is Alice and I work in machine learning.", None),
    ("What is gradient descent?", "It is an optimisation algorithm."),
    ("How does learning rate affect it?", "It controls the step size."),
    ("What is my name?", None),
]

for human_msg, ai_response in convs:
    ai_resp = ai_response or "OK, got it."
    buffer_memory.chat_memory.add_user_message(human_msg)
    buffer_memory.chat_memory.add_ai_message(ai_resp)

mem_vars = buffer_memory.load_memory_variables({})
history  = mem_vars["history"]

print(f"  ConversationBufferMemory: {len(history)} messages in history")
for msg in history:
    role    = "Human" if isinstance(msg, HumanMessage) else "AI"
    content = msg.content[:55] + ("..." if len(msg.content) > 55 else "")
    print(f"    [{role:<5}] {content}")
print()
print(f"  Total chars in history: {sum(len(m.content) for m in history)}")
print(f"  ⚠ Buffer memory grows unboundedly — hits context limit for long sessions")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: ConversationBufferWindowMemory (sliding window)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — ConversationBufferWindowMemory (k=3 turns)")
print("━" * 65)
print()

window_memory = ConversationBufferWindowMemory(k=3, return_messages=True,
                                                memory_key="history")

# Add 6 conversation turns
all_turns = [
    ("Hello, I am Bob.",                "Hi Bob!"),
    ("I am a data scientist.",          "Great to know!"),
    ("I work with neural networks.",    "Interesting field!"),
    ("What is my profession?",          "You're a data scientist."),
    ("What framework do I use?",        None),
]
for human, ai in all_turns:
    window_memory.chat_memory.add_user_message(human)
    window_memory.chat_memory.add_ai_message(ai or "I see.")

window_vars = window_memory.load_memory_variables({})
window_hist = window_vars["history"]

print(f"  Window memory (k=3): keeps only LAST 3 human+AI turn pairs")
print(f"  Total turns added: {len(all_turns)}")
print(f"  Messages in window: {len(window_hist)}")
for msg in window_hist:
    role    = "Human" if isinstance(msg, HumanMessage) else "AI"
    content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
    print(f"    [{role:<5}] {content}")
print()

# Compare memory sizes
print(f"  Memory comparison after {len(all_turns)} turns:")
print(f"    Buffer memory:  {sum(len(m.content) for m in history)} chars (grows forever)")
print(f"    Window memory:  {sum(len(m.content) for m in window_hist)} chars (bounded)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Modern LCEL memory with RunnableWithMessageHistory
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — RunnableWithMessageHistory (modern LCEL pattern)")
print("━" * 65)
print()

# Session store: session_id → ChatMessageHistory
session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# Build a simple conversational chain
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with memory."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

base_chain = conv_prompt | llm | StrOutputParser()

# Wrap with message history — this adds memory automatically
chain_with_memory = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key   = "input",
    history_messages_key = "history",
)

# Simulate two separate users with independent session histories
sessions = {
    "user_alice": [
        "Hello! My name is Alice and I'm a researcher.",
        "I study natural language processing.",
        "What is my name and what do I study?",
    ],
    "user_bob": [
        "Hi there. I'm Bob, a software engineer.",
        "What is my name?",
    ],
}

print(f"  RunnableWithMessageHistory: separate histories per session_id")
print()
for session_id, messages in sessions.items():
    print(f"  Session: {session_id}")
    for msg in messages:
        result = chain_with_memory.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"    H: {msg[:55]}")
        print(f"    A: {result.strip()[:70]}")
    history_len = len(session_store[session_id].messages)
    print(f"    (history: {history_len} messages stored for this session)")
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Tool creation patterns
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Tool creation: @tool decorator and StructuredTool")
print("━" * 65)
print()

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Supports +, -, *, /, ** (power), sqrt(), and standard Python math.
    Example: '2 + 2', 'sqrt(16)', '3**4 / 9'"""
    try:
        result = eval(expression, {"__builtins__": {}},
                      {"sqrt": math.sqrt, "pi": math.pi,
                       "log": math.log, "abs": abs, "round": round})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a given city.
    Returns temperature and weather description.
    unit: 'celsius' or 'fahrenheit'"""
    # Mock weather data
    weather_db = {
        "paris": {"temp_c": 18, "desc": "partly cloudy"},
        "london": {"temp_c": 12, "desc": "overcast"},
        "tokyo": {"temp_c": 25, "desc": "sunny"},
        "new york": {"temp_c": 20, "desc": "clear"},
    }
    city_key = city.lower()
    if city_key not in weather_db:
        return f"Weather data not available for {city}"
    data = weather_db[city_key]
    temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9/5 + 32
    unit_symbol = "°C" if unit == "celsius" else "°F"
    return f"{city}: {temp}{unit_symbol}, {data['desc']}"

@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web for current information on a topic.
    Returns a summary of the top search results."""
    # Mock search results
    results = {
        "transformer": "The Transformer architecture was introduced in 2017...",
        "langchain": "LangChain is an open-source framework for building LLM apps...",
        "openai": "OpenAI is an AI research company founded in 2015...",
        "python": "Python is a high-level programming language created by Guido van Rossum...",
    }
    for keyword, result in results.items():
        if keyword in query.lower():
            return f"Search results for '{query}': {result}"
    return f"Search results for '{query}': Found {max_results} relevant articles."

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol.
    ticker: Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"""
    mock_prices = {
        "AAPL": 185.50, "GOOGL": 142.30, "MSFT": 415.20,
        "NVDA": 875.40, "TSLA": 245.80,
    }
    price = mock_prices.get(ticker.upper())
    if price:
        return f"{ticker.upper()}: ${price:.2f} USD"
    return f"Ticker '{ticker}' not found."

tools = [calculator, get_weather, web_search, get_stock_price]

print(f"  Registered tools ({len(tools)}):")
for t in tools:
    print(f"  [{t.name:<18}] {t.description[:60]}")
    print(f"  {' '<20}  args: {list(t.args.keys())}")
print()

# Test individual tools
print(f"  Tool execution tests:")
print(f"    calculator('sqrt(144) + 3**2') → {calculator.invoke('sqrt(144) + 3**2')}")
print(f"    get_weather('Paris')           → {get_weather.invoke({'city': 'Paris'})}")
print(f"    get_stock_price('NVDA')        → {get_stock_price.invoke('NVDA')}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: Simulated Agent ReAct loop
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Agent ReAct loop: Thought → Action → Observation")
print("━" * 65)
print()

class SimpleReActAgent:
    """
    Simulates the ReAct (Reason + Act) agent loop.
    Demonstrates: multi-step reasoning, tool selection, observation integration.
    In production: use create_tool_calling_agent + AgentExecutor.
    """
    def __init__(self, tools):
        self.tools      = {t.name: t for t in tools}
        self.max_steps  = 5

    def run(self, question: str) -> str:
        print(f"  Question: {question}")
        print(f"  {'─'*55}")

        # Simulate the reasoning trace the agent would produce
        traces = self._get_trace(question)
        for step in traces:
            if step["type"] == "thought":
                print(f"  Thought: {step['content']}")
            elif step["type"] == "action":
                tool_name = step["tool"]
                tool_args = step["args"]
                print(f"  Action:  {tool_name}({tool_args})")
                # Execute the actual tool
                tool_fn = self.tools.get(tool_name)
                if tool_fn:
                    obs = tool_fn.invoke(tool_args)
                    print(f"  Observe: {obs}")
            elif step["type"] == "answer":
                print(f"  Answer:  {step['content']}")
                return step["content"]
        return "I was unable to find a complete answer."

    def _get_trace(self, question):
        q = question.lower()
        if "weather" in q and "temperature" in q:
            city = "Paris" if "paris" in q else "London" if "london" in q else "Tokyo"
            return [
                {"type": "thought",
                 "content": f"The user wants weather in {city}. I'll use get_weather."},
                {"type": "action",  "tool": "get_weather", "args": {"city": city}},
                {"type": "answer",
                 "content": f"Based on the weather data, I can provide current conditions for {city}."},
            ]
        elif "stock" in q or "price" in q:
            ticker = "NVDA" if "nvidia" in q else "AAPL" if "apple" in q else "MSFT"
            return [
                {"type": "thought",
                 "content": f"The user wants stock price. I'll look up {ticker}."},
                {"type": "action", "tool": "get_stock_price", "args": ticker},
                {"type": "answer",
                 "content": "Here is the current stock price information."},
            ]
        elif "calculate" in q or "math" in q or any(c in q for c in "+-*/^"):
            expr = "2 ** 10" if "1024" in q or "power" in q else "sqrt(256)"
            return [
                {"type": "thought",
                 "content": "The user needs a mathematical calculation."},
                {"type": "action", "tool": "calculator", "args": expr},
                {"type": "answer", "content": "The mathematical result is above."},
            ]
        else:
            return [
                {"type": "thought",
                 "content": "Let me search for relevant information."},
                {"type": "action", "tool": "web_search",
                 "args": {"query": question, "max_results": 3}},
                {"type": "answer",
                 "content": "Based on my search, here is what I found."},
            ]

agent = SimpleReActAgent(tools)

test_agent_queries = [
    "What is the current temperature in Paris?",
    "What is the stock price of NVIDIA?",
    "Calculate 2 to the power of 10",
    "What is LangChain?",
]

for query in test_agent_queries:
    agent.run(query)
    print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Production agent configuration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Production agent patterns and configuration")
print("━" * 65)
print()

AGENT_PATTERNS = """
# ── Production agent with tool calling (modern approach) ─────────────
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator, get_weather, web_search, get_stock_price]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools. "
               "Use tools when needed to answer questions accurately."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),   # where tool calls appear
])

agent    = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent                = agent,
    tools                = tools,
    verbose              = True,       # print the ReAct trace
    max_iterations       = 10,         # prevent infinite loops
    max_execution_time   = 30,         # seconds timeout
    handle_parsing_errors = True,      # gracefully handle malformed tool calls
    return_intermediate_steps = True,  # include tool call trace in output
)

result = executor.invoke({
    "input": "What is the weather in Tokyo and what is 15% of 285?",
})
print(result["output"])
# result["intermediate_steps"] = [(AgentAction, observation), ...]

# ── Adding memory to the agent ───────────────────────────────────────
agent_with_memory = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key   = "input",
    history_messages_key = "chat_history",
)
"""
print(AGENT_PATTERNS)

print(f"  Agent configuration guide:")
print(f"  ┌─────────────────────────────────────────────────────────────────┐")
print(f"  │ Setting                  │ Recommended  │ Why                    │")
print(f"  ├─────────────────────────────────────────────────────────────────┤")
print(f"  │ max_iterations           │ 5–10         │ Prevents infinite loops│")
print(f"  │ max_execution_time       │ 30–60s       │ Latency guardrail      │")
print(f"  │ handle_parsing_errors    │ True         │ Graceful degradation   │")
print(f"  │ return_intermediate_steps│ True         │ Debugging + logging    │")
print(f"  │ temperature              │ 0            │ Consistent tool calls  │")
print(f"  │ verbose                  │ False (prod) │ Reduce log noise       │")
print(f"  └─────────────────────────────────────────────────────────────────┘")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Production Patterns — Evaluation, Caching, Streaming & LangSmith": {
        "description": (
            "Production-hardening patterns for LangChain applications. "
            "LLM response caching: InMemoryCache and SQLiteCache. "
            "Token counting and cost tracking with callbacks. "
            "Streaming with .stream() and async .astream() patterns. "
            "Fallback chains for multi-provider resilience. "
            "Retry logic with exponential backoff. "
            "Chain evaluation: string evaluator and criteria evaluator. "
            "LangSmith tracing setup and trace inspection. "
            "LangServe deployment pattern for REST API. "
            "Advanced RAG: multi-query retrieval for higher recall. "
            "Complete production architecture diagram and checklist."
        ),
        "language": "python",
        "code": '''
import time
import json
import hashlib
import asyncio
from typing import Any, Iterator, AsyncIterator, Dict, List

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableParallel
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.globals import set_llm_cache
    from langchain_core.caches import InMemoryCache
    from langchain.callbacks import get_openai_callback
    print("  langchain-core + langchain: installed")
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'langchain', 'langchain-core', '--quiet'], check=True)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableParallel
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.globals import set_llm_cache
    from langchain_core.caches import InMemoryCache

print("=" * 65)
print("  PRODUCTION PATTERNS — CACHING, STREAMING & LANGSMITH")
print("=" * 65)
print()

# ── Instrumented mock LLM with latency simulation ────────────────────
from pydantic import Field

class InstrumentedMockLLM(BaseChatModel):
    model_name: str  = "mock-instrumented"
    latency_ms: float = 50.0
    call_log: list = Field(default_factory=list)
    token_counter: dict = Field(default_factory=lambda: {"input": 0, "output": 0})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.latency_ms / 1000)
        prompt_text = " ".join(
            m.content for m in messages if hasattr(m, "content")
        )
        # Simulate token counting
        input_tokens  = len(prompt_text.split())
        output_tokens = 30
        self.token_counter["input"]  += input_tokens
        self.token_counter["output"] += output_tokens
        self.call_log.append({
            "prompt_tokens":    input_tokens,
            "completion_tokens": output_tokens,
            "latency_ms":       self.latency_ms,
        })
        content = self._respond(prompt_text.lower())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk
        content = self._generate(messages).generations[0].message.content
        words   = content.split()
        for i, word in enumerate(words):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + (" " if i < len(words)-1 else "")))
            time.sleep(0.01)

    def _respond(self, text):
        if "france" in text: return "The capital of France is Paris, a city with rich history and culture."
        if "python" in text: return "Python is an interpreted, high-level programming language known for readability."
        if "machine learning" in text: return "Machine learning is a subset of AI that learns from data through algorithms."
        return "This is a thoughtful response to your query, based on careful analysis of the available information."

    def reset_counters(self):
        self.call_log.clear()
        self.token_counter = {"input": 0, "output": 0}

    @property
    def _llm_type(self): return "mock-instrumented"
    @property
    def _identifying_params(self): return {}

llm = InstrumentedMockLLM(latency_ms=80)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: LLM Response Caching
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — LLM caching: InMemoryCache and SQLiteCache")
print("━" * 65)
print()

# InMemoryCache: caches by (prompt_hash, llm_params) → response
cache = InMemoryCache()
set_llm_cache(cache)

prompt = ChatPromptTemplate.from_template("{question}")
chain  = prompt | llm | StrOutputParser()

queries = [
    "What is the capital of France?",
    "What is Python programming language?",
    "What is the capital of France?",  # duplicate — should hit cache
    "What is Python programming language?",   # duplicate — cache hit
    "What is machine learning?",
]

print(f"  Testing LLM cache (5 queries, 2 duplicates):")
print(f"  {'Query':<43} {'Time (ms)':>10} {'Cached':>8}")
print(f"  {'─'*65}")
llm.reset_counters()

results_and_times = []
for q in queries:
    t0  = time.perf_counter()
    res = chain.invoke({"question": q})
    ms  = (time.perf_counter() - t0) * 1000
    results_and_times.append((q, ms))

call_count_after = len(llm.call_log)
for i, (q, ms) in enumerate(results_and_times):
    cached = "✓ HIT" if (i >= 2 and queries[i] in [queries[j] for j in range(i)]) else "MISS"
    print(f"  {q[:42]:<43} {ms:>10.1f} {cached:>8}")

print()
print(f"  Total LLM calls made:    {call_count_after}  (5 queries - 2 cached = {5-2} actual)")
print(f"  Tokens consumed:         input={llm.token_counter['input']} output={llm.token_counter['output']}")
print()

CACHE_REFERENCE = """
# Cache options in LangChain:

# 1. In-memory (process-local, not persistent)
from langchain_core.caches import InMemoryCache
set_llm_cache(InMemoryCache())

# 2. SQLite (persistent, single-machine)
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 3. Redis (distributed, production scale)
from langchain_community.cache import RedisCache
import redis
set_llm_cache(RedisCache(redis_=redis.from_url("redis://localhost:6379"),
                          ttl=3600))  # 1-hour TTL

# 4. Semantic cache (cache similar queries, not exact matches)
from langchain_community.cache import RedisSemanticCache
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.2,  # cosine similarity threshold
))
"""
print(CACHE_REFERENCE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Streaming — real-time token delivery
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Streaming: .stream() and .astream()")
print("━" * 65)
print()

set_llm_cache(InMemoryCache())   # fresh cache for streaming test

stream_chain = (
    ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
    | llm
    | StrOutputParser()
)

print(f"  Synchronous .stream() — tokens appear as generated:")
print(f"  ", end="")
token_times = []
t_start = time.perf_counter()
for chunk in stream_chain.stream({"topic": "machine learning"}):
    token_times.append(time.perf_counter() - t_start)
    print(chunk, end="", flush=True)
print()
print()

ttft = token_times[0] * 1000 if token_times else 0
total_t = token_times[-1] * 1000 if token_times else 0
print(f"  Time to first token:  {ttft:.0f}ms")
print(f"  Total streaming time: {total_t:.0f}ms")
print(f"  Tokens streamed:      {len(token_times)}")
print()

# Async streaming pattern
async def astream_demo():
    chunks_received = []
    async for chunk in stream_chain.astream({"topic": "Python"}):
        chunks_received.append(chunk)
    return "".join(chunks_received)

print(f"  Async .astream() pattern (production recommended):")
async_result = asyncio.get_event_loop().run_until_complete(astream_demo())
print(f"  Result: '{async_result[:70]}...'")
print()

STREAMING_PATTERNS = """
# ── Production streaming patterns ────────────────────────────────────

# 1. FastAPI SSE endpoint with streaming
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def stream_chat(question: str):
    async def generate():
        async for chunk in chain.astream({"question": question}):
            yield f"data: {json.dumps({'token': chunk})}\\n\\n"
        yield "data: [DONE]\\n\\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

# 2. Collect stream into string (when you need the full result)
full_response = ""
for chunk in chain.stream(input):
    full_response += chunk
    print(chunk, end="", flush=True)   # show to user in real-time

# 3. Stream with intermediate steps (for agents)
async for event in agent_executor.astream_events({"input": query}, version="v1"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
"""
print(STREAMING_PATTERNS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Fallbacks and retry logic
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Fallbacks and retry for production resilience")
print("━" * 65)
print()

# Simulate a flaky primary LLM
class FlakyLLM(BaseChatModel):
    fail_count: int = 0
    max_fails:  int = 2

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.fail_count += 1
        if self.fail_count <= self.max_fails:
            raise Exception(f"Rate limit exceeded (simulated fail {self.fail_count})")
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="Primary LLM response (after failures)"))
        ])
    @property
    def _llm_type(self): return "flaky"
    @property
    def _identifying_params(self): return {}

class ReliableLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="Fallback LLM response (reliable backup)"))
        ])
    @property
    def _llm_type(self): return "reliable"
    @property
    def _identifying_params(self): return {}

primary_llm  = FlakyLLM(max_fails=2)
fallback_llm = ReliableLLM()

prompt = ChatPromptTemplate.from_template("Answer: {question}")

primary_chain  = prompt | primary_llm  | StrOutputParser()
fallback_chain = prompt | fallback_llm | StrOutputParser()

# Chain with automatic fallback
robust_chain = primary_chain.with_fallbacks([fallback_chain])

print(f"  Fallback chain test (primary fails 2 times, then fallback used):")
for attempt in range(4):
    try:
        result = robust_chain.invoke({"question": "What is AI?"})
        print(f"    Attempt {attempt+1}: SUCCESS → '{result[:50]}'")
    except Exception as e:
        print(f"    Attempt {attempt+1}: ERROR   → {e}")
print()

# Retry pattern
RETRY_PATTERN = """
# Retry with exponential backoff (for transient failures)
robust_with_retry = chain.with_retry(
    stop_after_attempt       = 3,
    wait_exponential_jitter  = True,       # adds random jitter to prevent thundering herd
    retry_if_exception_type  = (Exception,)  # retry on these exception types
)

# Combine fallbacks AND retry:
ultra_robust = (chain
    .with_retry(stop_after_attempt=2)
    .with_fallbacks([backup_chain.with_retry(stop_after_attempt=2)]))
"""
print(RETRY_PATTERN)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Async batch processing for throughput
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Async batch: high-throughput parallel processing")
print("━" * 65)
print()

test_questions = [
    {"question": "What is machine learning?"},
    {"question": "What is the capital of France?"},
    {"question": "What is Python programming language?"},
    {"question": "Explain machine learning simply."},
]

# Synchronous batch (sequential)
sync_llm = InstrumentedMockLLM(latency_ms=50)
set_llm_cache(InMemoryCache())  # clear cache
sync_chain = ChatPromptTemplate.from_template("{question}") | sync_llm | StrOutputParser()

t0 = time.perf_counter()
sync_results = sync_chain.batch(test_questions)
t_sync = (time.perf_counter() - t0) * 1000

# Async batch (parallel)
async_llm = InstrumentedMockLLM(latency_ms=50)
async_chain = ChatPromptTemplate.from_template("{question}") | async_llm | StrOutputParser()

async def run_async_batch():
    return await async_chain.abatch(test_questions, config={"max_concurrency": 4})

t0 = time.perf_counter()
async_results = asyncio.get_event_loop().run_until_complete(run_async_batch())
t_async = (time.perf_counter() - t0) * 1000

print(f"  Batch processing {len(test_questions)} queries (50ms each):")
print(f"  {'─'*45}")
print(f"    Sync .batch():   {t_sync:7.0f}ms  (sequential execution)")
print(f"    Async .abatch(): {t_async:7.0f}ms  (parallel execution)")
if t_async > 0:
    print(f"    Speedup:         {t_sync/t_async:.1f}×  (max theoretical: {len(test_questions)}×)")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 5: LangSmith setup and tracing
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — LangSmith: observability and tracing setup")
print("━" * 65)
print()

LANGSMITH_SETUP = """
# ── LangSmith setup (add to environment before importing langchain) ───

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = "lsv2_pt_..."     # from smith.langchain.com
os.environ["LANGCHAIN_PROJECT"]    = "my-rag-app"      # groups traces by project
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

# After setting env vars, ALL chain.invoke() calls are automatically traced.
# No code changes required. Every invocation records:
#   - Full input and output at every component
#   - Latency per step and end-to-end
#   - Token counts and estimated API cost
#   - Error traces with full stack
#   - Nested call graph (chain → prompt → LLM → parser → ...)

chain = prompt | llm | StrOutputParser()

# Add custom metadata to a trace:
result = chain.invoke(
    {"question": "What is RAG?"},
    config={
        "metadata": {
            "user_id":      "user_123",
            "session_id":   "sess_456",
            "app_version":  "2.1.0",
            "feature_flag": "rag_v2",
        },
        "tags": ["production", "rag-pipeline"],
        "run_name": "rag-qa-chain",
    }
)

# ── Evaluating with LangSmith datasets ───────────────────────────────
from langsmith import Client
client = Client()

# Create a dataset from example Q&A pairs
dataset = client.create_dataset("rag-eval-v1", description="RAG evaluation set")
examples = [
    {"inputs": {"question": "What is BERT?"},
     "outputs": {"answer": "BERT is a bidirectional transformer encoder."}},
    {"inputs": {"question": "Who invented the Transformer?"},
     "outputs": {"answer": "Vaswani et al. at Google Brain in 2017."}},
]
client.create_examples(inputs=[e["inputs"] for e in examples],
                        outputs=[e["outputs"] for e in examples],
                        dataset_id=dataset.id)

# Run evaluation — uses LLM-as-judge for correctness
from langsmith.evaluation import evaluate, LangChainStringEvaluator

results = evaluate(
    chain.invoke,
    data        = "rag-eval-v1",
    evaluators  = [LangChainStringEvaluator("cot_qa")],  # chain-of-thought QA
    experiment_prefix = "rag-v2",
)
# Results appear in the LangSmith UI with pass rates per evaluator
"""
print(LANGSMITH_SETUP)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 6: Complete production architecture
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 6 — Complete production architecture and checklist")
print("━" * 65)
print()

PRODUCTION_ARCH = """
Complete Production RAG Architecture:

                         ┌─────────────────────────────────────┐
INGESTION PIPELINE       │                                     │
(Offline / Batch)        │  1. Load documents                  │
                         │     (PDF, web, DB, API)             │
                         │  2. Split into chunks               │
                         │     (RecursiveCharacterTextSplitter) │
                         │  3. Embed chunks                    │
                         │     (text-embedding-3-small)        │
                         │  4. Store in vector DB              │
                         │     (FAISS / Chroma / Pinecone)     │
                         └────────────────┬────────────────────┘
                                          │
                         ┌────────────────▼────────────────────┐
QUERY PIPELINE           │                                     │
(Online / Per-request)   │  5. Receive user question           │
                         │  6. [Optional] Rewrite with history │
                         │  7. Embed query                     │
                         │  8. Retrieve top-k chunks (MMR)     │
                         │  9. [Optional] Rerank (Cohere)      │
                         │  10. Build prompt with context      │
                         │  11. LLM generation (streaming)     │
                         │  12. Return answer + sources        │
                         └────────────────┬────────────────────┘
                                          │
                         ┌────────────────▼────────────────────┐
OBSERVABILITY            │                                     │
(Always-on)              │  LangSmith traces every call        │
                         │  Latency, tokens, cost per step     │
                         │  Automated evaluations on datasets  │
                         └─────────────────────────────────────┘
"""
print(PRODUCTION_ARCH)

print("  Production checklist:")
checklist = [
    ("Streaming",          "Use .astream() for all user-facing responses"),
    ("Caching",            "SQLiteCache or Redis for repeated identical queries"),
    ("Fallbacks",          ".with_fallbacks([backup_provider]) for uptime"),
    ("Retry logic",        ".with_retry(stop_after_attempt=3) for transient errors"),
    ("Max iterations",     "Set max_iterations=10 on all agents"),
    ("Async batch",        "Use .abatch(max_concurrency=10) for bulk processing"),
    ("LangSmith tracing",  "Set LANGCHAIN_TRACING_V2=true in production"),
    ("Cost tracking",      "Monitor token counts; set alert thresholds"),
    ("Chunk tuning",       "Benchmark chunk_size/overlap for your specific corpus"),
    ("MMR retrieval",      "Use MMR over simple similarity to reduce redundancy"),
    ("Re-ranking",         "Add Cohere/cross-encoder reranker for quality boost"),
    ("Question rewrite",   "Rewrite question with history before retrieval"),
]
for item, desc in checklist:
    print(f"    ✓  {item:<22} {desc}")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }