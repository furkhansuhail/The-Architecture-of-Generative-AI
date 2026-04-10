"""
TOPIC TEMPLATE
==============
Copy this file and rename it:  NN_topic_name.py
Fill in each section. The app will auto-discover this module.

Naming convention:
  01_tokenization_embeddings.py
  02_language_modeling.py
  ...
"""

# ── Display name (shown in sidebar and as page title) ──────────────────────
TOPIC_NAME   = "Knowledge Graph"
DISPLAY_NAME = "RAG Knowledge Graph"
ICON         = "📖"
SUBTITLE     = "Knowledge Graph utility in RAG"

# ── Theory ─────────────────────────────────────────────────────────────────
THEORY = """

##### What is a Knowledge Graph ?

At its core, a knowledge graph is a way of storing and representing information as a 
network of connected entities — things, people, places, concepts — and the relationships 
between them. 
Rather than storing data in rows and columns like a spreadsheet, it stores data as a 
web of nodes and edges.
Think of it like a map of facts where every fact is a triple: Subject → Relationship → Object. For example:

    Leonardo da Vinci → painted → Mona Lisa
    Mona Lisa         → is displayed at → The Louvre
    The Louvre        → is located in → Paris


The Core Building Block: The Triple

Everything in a knowledge graph is expressed as a triple (also called a "fact" or "statement"):

    Subject → Predicate (relationship) → Object

This is sometimes written as (S, P, O). The relationship is the edge connecting two nodes. 

There are two types of objects:

    * Entity nodes   — things in the world (a person, a city, a product)
    * Literal values — data like text, numbers, or dates (da Vinci's birth year is 1452)

How it Differs from a Regular Database

A traditional relational database stores data in rigid tables. 
If you want to connect "artists" to "artworks" to "museums," you need multiple tables
and complex JOIN queries. Adding a new type of connection (say, "influenced by") means 
altering your schema.

A knowledge graph is fundamentally different in two ways. 

First, it's schema-flexible — you can add new entity types and relationships at any 
       time without restructuring everything. 
       
Second, it's built for traversal — the whole point is to follow chains of relationships 
        to answer questions, not just look up rows.


## What Are They Used For?

Search engines — Google's Knowledge Graph powers the info boxes you see when you search for a person or place. 
                 It connects millions of entities so results can say "people also search for..." intelligently.
                 
Recommendation systems — Netflix and Spotify use graph-like structures to connect 
                         users → preferences → content → genres → similar content, 
                         enabling "because you watched X" suggestions.
                         
Virtual assistants — When you ask Siri "Who directed the movie that won Best Picture in 2020?", 
                     the system must traverse several hops across entities to get you the answer. 
                     A knowledge graph makes this possible.
                     
Healthcare & drug discovery — Medical knowledge graphs connect 
                              diseases → symptoms → drugs → side effects → genes. 
                              Researchers can query relationships like "which drugs interact with this protein?" 
                              across billions of facts.
                              
Fraud detection — Financial companies build graphs of transactions, accounts, and people. 
                  Fraud patterns become visible as suspicious clusters or unusual connection 
                  paths that would be impossible to spot in flat data.
                  
Enterprise knowledge management — Companies use knowledge graphs to connect 
                                  employees, projects, documents, and expertise, so asking "who in this company 
                                  has worked on X?" becomes answerable.
                                  
## Key Concepts to Know: 

Ontology — The "rules" of the graph. It defines what kinds of entities exist (Person, Place, Event) 
           and what relationships are valid between them. It's the vocabulary the graph speaks.
           
Inference — One of the most powerful features. 
            If the graph knows "Paris is in France" and "France is in Europe", it can automatically 
            infer "Paris is in Europe" — even if that fact was never explicitly stored.
            
SPARQL — The query language for knowledge graphs (like SQL for databases). 
         You use it to ask graph-structured questions like "find all museums in cities where a 
         Renaissance painter was born."
         
RDF (Resource Description Framework) — The standard data format for encoding knowledge graphs 
                                       on the web, where every entity and relationship is 
                                       identified by a unique web URL.          


In short, knowledge graphs shine whenever your data is highly interconnected, 
the relationships between things are the insight, or you need to answer questions 
that require reasoning across many hops of context. 
They're the backbone of modern AI systems that need to "know things about the world."


## Breakdown of Knowledge Graph-Enhanced RAG (GraphRAG)

Before we dive in, a quick recap: standard RAG works by converting your question into a vector embedding, 
searching a database of document chunks for similar vectors, and handing those chunks to the LLM as context. 
It's powerful, but it's fundamentally a similarity search — it finds passages that look like your question, 
not necessarily ones that answer it through reasoning.

Plugging a knowledge graph into that pipeline changes the retrieval step from "find similar text" to 
"traverse a network of facts." 

This is often called GraphRAG.

Here's the overall pipeline:

                         +-----------------------+
                         |       User query      |
                         | Natural lang question |
                         +-----------+-----------+
                                     |
                    _________________V_________________
                   |                                   |
        +----------V-----------+            +----------V-----------+
        |   Entity extraction  |            |    Query embedding   |
        | Find nouns, concepts |            | Convert query vector |
        +----------+-----------+            +----------+-----------+
                   |                                   |
        +----------V-----------+            +----------V-----------+
        |    Graph traversal   |            |     Vector search    |
        | Hop across links     |            | Find similar chunks  |
        +----------^-----------+            +----------^-----------+
                   :                                   :
                   :                                   : 
        +----------+-----------+            +----------+-----------+
        |    Knowledge graph   |            |    Vector database   |
        | Entities + relations |            | Embedded doc chunks  |
        +----------+-----------+            +----------+-----------+
                   |                                   |
                   |        +------------------+       |
                   +------->|  Context builder |<------+
                            | Merge facts/pass |
                            +--------+---------+
                                     |
                            +--------V---------+
                            |        LLM       |
                            | Grounded answer  |
                            +------------------+
    
      LEGEND:
      |  Solid line  = Flow/Step connection
      :  Dotted line = Query to data store
      

Break down what's happening at each stage.

# **Stage 1 - Entity Extraction**

When a query like "What drugs does Dr. Sarah Chen prescribe for Type 2 diabetes?" comes in, 
the system doesn't immediately search for it. First it runs entity extraction — a process 
(usually a smaller NLP model) that identifies the meaningful nouns and concepts in the question:

    * Dr. Sarah Chen → entity of type Person/Doctor
    * Type 2 diabetes → entity of type Disease
    * drugs → the relationship being asked about (prescribes)

These extracted entities become the entry points — the starting nodes — for navigating the graph. 
Without this step, you have no foothold in the graph to begin traversal from.

# **Stage 2 — Graph Traversal (the core integration)**

This is where the knowledge graph is actually used. 

Once you have your entry-point entities, 
the system performs a multi-hop traversal — following edges outward from those 
nodes to collect related facts.
        
Text: 
    
    "Dr. Sarah Chen works at City Hospital. She specializes in Endocrinology. 
    She prescribes Metformin and Ozempic for Type 2 Diabetes. 
    Metformin side effects include GI upset. 
    Metformin interactions: avoid alcohol."
        
                [ Hop 0 ]                [ Hop 0 ]
          +-------------------+    +-------------------+
          |   Dr. Sarah Chen  |    |  Type 2 diabetes  |
          |    Entry point    |    |    Entry point    |
          +--------+----------+    +------------+------+
                   |                            |
            _______|_______              _______|_______
           |               |            |               |
        [ Hop 1 ]       [ Hop 1 ]   [ Hop 1 ]       [ Hop 1 ]
        +----------+    +---------+ +----------+    +---------+
        | City Hosp|    | Endocrin| | Insul Res|    | Pancreas|
        | works at |    | special | | mechanis |    |  organ  |
        +----+-----+    +---------+ +----+-----+    +---------+
             |                           |
          ___|_______                  ______|______
         |           |                 |            |
        [ Hop 2 ] [ Hop 2 ]       [ Hop 2 ]   [ Hop 2 ]
        +---------+ +---------+   +---------+ +---------+
        | Metform | | Ozempic |   | Side Eff| | Drug Int|
        | prescri | | prescri |   | GI upset| | no alcoh|
        +---------+ +---------+   +---------+ +---------+
        
        LEGEND:
        [ Hop 0 ] = Entry node (from query)
        [ Hop 1 ] = Hop 1 facts
        [ Hop 2 ] = Hop 2 facts (deeper context)


Each hop outward collects a richer, structured web of context. 
The system typically stops at 2–3 hops — going deeper gets exponentially larger and less relevant. 
All the retrieved triples are then serialized into plain text facts that the LLM can read:


# **Stage 3 — Parallel Vector Search**

Simultaneously with graph traversal, the system also runs a standard vector similarity 
search on the document store. 
This retrieves raw text passages — like paragraphs from clinical guidelines, research papers, 
or internal docs — that are semantically similar to the query.

These two retrieval paths are complementary. Graph traversal gives you structured, 
precise relational facts. Vector search gives you unstructured, nuanced prose. 
Neither alone is as powerful as both together.


# **Stage 4 — Context Merging & Generation**

The context builder combines both retrieved results into a single prompt that gets sent to the LLM. 

It typically looks like this:

    [GRAPH FACTS]
    - Dr. Chen specializes in Endocrinology
    - Dr. Chen prescribes: Metformin, Ozempic
    - Metformin contraindicated with: alcohol, kidney disease
    
    [RETRIEVED PASSAGES]
    "...GLP-1 receptor agonists like semaglutide (Ozempic) have 
    shown significant HbA1c reduction in clinical trials..."
    
    [QUESTION]
    What drugs does Dr. Sarah Chen prescribe for Type 2 diabetes?

The LLM then generates an answer that is grounded in both the structured facts and the retrieved prose, 
rather than relying on its own potentially outdated or hallucinated knowledge.


Why This Is Better Than Standard RAG: 

Standard RAG would retrieve a paragraph that mentions diabetes drugs in general. 
It might miss Dr. Chen entirely if her name doesn't appear in any single relevant chunk. 
GraphRAG fixes three specific weaknesses:

    Multi-hop reasoning — "What city is the hospital where the doctor who treats this disease works in?" 
                          Standard RAG can't reliably chain across that many documents. 
                          Graph traversal does it natively in one query.
                          
                          
    Precise relational facts — The graph stores the relationship 
                               (Dr. Chen) –[prescribes]→ (Metformin) as a discrete, 
                               queryable fact — not buried in paragraph 4 of a 5-page document. 
                               
   Reduced hallucination — Because the LLM receives explicit, structured facts alongside prose, 
                           it has less room to "fill in the gaps" with confabulated information. 
                           The facts act as hard constraints on the answer.
                           
The trade-off is complexity: you need to build and maintain the knowledge graph, 
keep it in sync with your document store, and tune how many hops to traverse per query. 

But for domains where accuracy and reasoning chains matter — medicine, law, finance, enterprise knowledge — 
the payoff is substantial. 


### Exploration 1:

Do knowledge graphs encode semantic relationships using vector-space geometry — similar to dot products or cosine 
similarity between embeddings?

For example, if "male" and "female" are represented as vectors at distances ~3.0 and ~3.5 from 
the origin respectively, can we infer that semantically related terms like "father" and "mother" 
would appear at proportionally similar distances — perhaps in an opposing or transformed 
direction — preserving the relational structure (e.g., male → father mirrors female → mother) ?

Additionally, could you write an interactive script that:

    * Takes a text input (paragraph or sentence)
    * Extracts entities and their relationships to construct a knowledge graph
    * Renders it visually with nodes and edges
    * Allows the user to hover over each node to inspect the underlying data 
    (entity type, relationships, vector coordinates, etc.)


This version makes the word embedding / vector space angle explicit 
(which is likely what you were thinking of), and clearly separates the conceptual question from the 
implementation request.


### Explanation for Exploration 1:

Knowledge graphs and vector embeddings are two different systems — 
but they can be combined, and that combination is one of the 
most exciting areas in modern AI.

A pure knowledge graph is symbolic. It stores (male) –[opposite_gender]→ (female) as a 
discrete logical triple. There's no geometry, no distance, no arithmetic. 
It knows the fact but has no sense of similarity.

What you're describing — the famous king − man + woman ≈ queen analogy — is Word2Vec / GloVe embeddings.
In those systems, the relational structure you described is real and measurable: the vector offset 
father − male ≈ mother − female holds because the embedding space learned that "gender" is a 
consistent geometric direction across semantic pairs. Your intuition about distances (~3.0, ~3.5 from origin) 
and opposing/transformed directions is exactly how TransE, one of the first knowledge graph embedding models, 
works — it literally models a relation r as a translation vector, so that head + r ≈ tail.

The key spectrum looks like this:

    +-----------------------+-----------------------+-----------+-----------+
    | System                | Representation        | Analogy   | Symbolic  |
    |                       |                       | reasoning | lookup    |
    +=======================+=======================+===========+===========+
    | Pure KG               | Symbolic triples      | No        | Yes       |
    +-----------------------+-----------------------+-----------+-----------+
    | Word2Vec/GloVe        | Dense vectors         | Yes       | No        |
    +-----------------------+-----------------------+-----------+-----------+
    | KG Embeddings (TransE,| Entities + relations  | Yes       | Approx-   |
    | RotatE)               | as vectors            |           | imate     |
    +-----------------------+-----------------------+-----------+-----------+
    | GraphRAG              | Both                  | Yes       | Yes       |
    +-----------------------+-----------------------+-----------+-----------+

Modern systems like GraphRAG live at the bottom row — they keep the symbolic graph for precise traversal, 
and use embeddings to find entry points and measure semantic proximity.

## **On the vector coordinates shown:** 

The coordinates displayed on hover are derived from the force simulation's final positions — 
which is actually a surprisingly meaningful proxy for semantic structure. 
The D3 force layout minimizes an energy function where connected nodes attract and unconnected 
nodes repel, so spatially close nodes genuinely tend to be semantically related. 
It's not Word2Vec, but the clustering geometry is structurally analogous.

For the real version of what you described
— the father − male ≈ mother − female arithmetic — that lives in knowledge graph embedding models. 

The three main approaches each encode it differently:

    * TransE models a relation as a translation:
        
        h + r = t, so (male) + (gender_flip) ≈ (female) and (father) + (gender_flip) ≈ (mother). 
    
    Your intuition about proportional distances in opposing directions is exactly this.
    

    * RotatE uses complex-number rotation instead of translation, which handles symmetric 
      and antisymmetric relations that TransE struggles with (e.g. sibling is symmetric; parent_of is not).
    
    * ComplEx / DistMult use bilinear scoring — the score of a triple (h, r, t) is a dot product in 
      complex space, which is precisely the cosine-similarity-style geometry you asked about.
    
So to directly answer your question: a pure knowledge graph does not have this property. 
But once you train a KG embedding on top of it, 
yes — the relational structure you described (gender offset preserving proportional distances, 
father/mother mirroring male/female) emerges as a learnable geometric structure in the embedding space.


**What is TransE (Translating Embeddings):**

TransE is a knowledge graph embedding model published by Bordes et al. in 2013. 
Its core idea is disarmingly simple: a relationship between two entities should 
look like a vector translation in space.

**The core equation**

For every triple (head, relation, tail) in the knowledge graph, 

TransE tries to learn vectors h, r, and t such that:

    h + r ≈ t
    head + relation ≈ tail 
    
That's it. Head entity plus relation vector should land you approximately on the tail entity. 
If the triple is true, the vectors should satisfy this. 
If it's false, they shouldn't.

**A concrete example**

Take the true fact: (Paris, capital_of, France)

TransE learns three vectors:

    h = the vector for Paris
    r = the vector for the relation capital_of
    t = the vector for France


After training, it should hold that Paris + capital_of ≈ France. 
The relation vector capital_of is literally an arrow in space — a direction and magnitude that, 
when added to any capital city, lands you near its country.

This generalizes. If the model has also seen (Berlin, capital_of, Germany) and 
(Rome, capital_of, Italy), the same capital_of vector should work for all of them. 
The model is forced to learn a consistent geometric meaning for every relation.


 
     y ^
       |      [Father] - - r=gender - -> [Mother]
       |         ^                          ^
       |  parent |                          | parent
       |         |                          |
       |      [Male]  - - r=gender - -> [Female]
       |
       |                                  [France]
       |                                   ^
       |                [Paris] ----------/ r=capital_of
       |                                 /
       |                                /      [Germany]
       |                [Berlin] ------/        ^
       |                              /        /
       |                             /--------/ r=capital_of
       |
       +------------------------------------------------------> x
                                                                   
       KEY:
       - [Node]   : Entity (Head or Tail)
       - ------>  : r (Relation vector)
       -  - - ->  : Parallel relation mapping

The diagram shows the key insight: the same relation vector r is reused across all triples of that type. 
Paris→France and Berlin→Germany use the identical capital_of arrow. 
This forces the model to learn a consistent geometric meaning, not just memorize individual pairs.


Core Concepts Visualized

    * Vector Translation (h + r ≈ t):
      The relationship "capital of" is represented by a specific vector (r). 
      When you add that vector to the coordinates of Paris (h), 
      you should land near the coordinates for France (t).
      
      Relational Parallelism: Notice that the vector for Paris → France is the same as Berlin → Germany. 
      This allows the system to perform analogy reasoning (e.g., "Paris is to France as Berlin is to...").
      
      Manifold Structure: 
      Entities with similar characteristics (like "Male" and "Female" or "Father" and "Mother") 
      are clustered in specific areas of the vector space, while their shared relationships 
      (gender, parentage) maintain consistent geometric distances.
      
      
**How training works**

TransE defines a scoring function for a triple:

    score(h, r, t) = −‖ h + r − t ‖
    
A true triple scores high (the distance is small). A false triple scores low. 
Training uses negative sampling — for every true triple, it corrupts it 
(randomly swaps head or tail with a random entity) to create a false triple, 
then optimizes so that:

    ‖ h + r − t ‖  (true triple)  <  ‖ h' + r − t' ‖  (false triple)
    
This is trained with a margin-based loss, pushing true triples below a 
threshold and false ones above it.


## **What TransE is good at — and where it breaks**

TransE works beautifully for one-to-one relations like capital_of or born_in. 
But it has well-known failure modes:

    Symmetric relations — If A is_married_to B then B is_married_to A. 
                          TransE can't model this because h + r = t and t + r = h 
                          can't both be true unless r = 0.
                          
    One-to-many relations — Shakespeare wrote_by Hamlet and Shakespeare wrote_by 
                            Macbeth can't both hold under h + r = t with the same h and r, 
                            because t would have to be in two places at once.
                            
    Composition — Reasoning like "uncle = parent's sibling" requires combining two 
                  relation vectors, which TransE doesn't explicitly model.
                  
These limitations led directly to successor models: 

    RotatE (relations as complex rotations, handles symmetry), 
    TransR (separate vector spaces per relation, handles one-to-many), and 
    ComplEx (complex-valued dot products, handles antisymmetry). 

But TransE remains the conceptual foundation they all build on — and for simple, 
one-to-one relational facts, it's still remarkably effective and fast to train.


### **Where the Knowledge Graph Lives in a RAG Model**

    ╔══════════════════════════════════════════════════════════════════╗
    ║                        BUILD TIME  (offline)                     ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   Raw Documents                                                  ║
    ║   (PDFs, articles,                                               ║
    ║    databases, etc.)                                              ║
    ║         │                                                        ║
    ║         ├─────────────────────────┐                              ║
    ║         │                         │                              ║
    ║         ▼                         ▼                              ║
    ║   ┌───────────────┐       ┌────────────────────┐                 ║
    ║   │  Chunk + embed│       │  Entity & relation │                 ║
    ║   │  into vectors │       │  extraction (NLP)  │                 ║
    ║   └───────┬───────┘       └────────┬───────────┘                 ║
    ║           │                        │                             ║
    ║           ▼                        ▼                             ║
    ║   ┌───────────────┐       ┌───────────────────┐                  ║
    ║   │  Vector store │       │  Knowledge graph  │                  ║
    ║   │  (Pinecone,   │       │  (Neo4j, Neptune, │                  ║
    ║   │   pgvector)   │       │   in-memory)      │                  ║
    ║   └───────────────┘       └───────────────────┘                  ║
    ║        [Store A]               [Store B]  ◄── lives here         ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    ╔══════════════════════════════════════════════════════════════════╗
    ║                       QUERY TIME  (live)                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║   User: "What drugs did Dr. Chen prescribe for diabetes?"        ║
    ║         │                                                        ║
    ║         ▼                                                        ║
    ║   ┌─────────────────┐                                            ║
    ║   │  Query analyzer │  ← splits the work into two lanes          ║
    ║   └────────┬────────┘                                            ║
    ║            │                                                     ║
    ║     ┌──────┴──────┐                                              ║
    ║     │             │                                              ║
    ║     ▼             ▼                                              ║
    ║  [Lane 1]      [Lane 2]                                          ║
    ║                                                                  ║
    ║  Embed query   Extract entities                                  ║
    ║  as vector     ("Dr. Chen",                                      ║
    ║     │           "diabetes")                                      ║
    ║     │               │                                            ║
    ║     ▼               ▼                                            ║
    ║  Vector store   Knowledge graph  ◄── queried here too            ║
    ║  similarity     graph traversal                                  ║
    ║  search         (2-3 hops out)                                   ║
    ║     │               │                                            ║
    ║     ▼               ▼                                            ║
    ║  Top-K text     Structured facts                                 ║
    ║  passages       (triples)                                        ║
    ║     │               │                                            ║
    ║     └──────┬────────┘                                            ║
    ║            │                                                     ║
    ║            ▼                                                     ║
    ║   ┌─────────────────────────────────────────┐                    ║
    ║   │           Context builder               │                    ║
    ║   │                                         │                    ║
    ║   │  [GRAPH FACTS]          [PASSAGES]      │                    ║
    ║   │  Dr. Chen → prescribes  "...GLP-1       │                    ║
    ║   │  → Metformin            agonists are    │                    ║
    ║   │  Metformin → treats     first-line..."  │                    ║
    ║   │  → Type 2 diabetes                      │                    ║
    ║   └──────────────────┬──────────────────────┘                    ║
    ║                      │                                           ║
    ║                      ▼                                           ║
    ║             ┌────────────────┐                                   ║
    ║             │      LLM       │                                   ║
    ║             │  (GPT, Claude, │                                   ║
    ║             │   Llama, etc.) │                                   ║
    ║             └───────┬────────┘                                   ║
    ║                     │                                            ║
    ║                     ▼                                            ║
    ║         Grounded, accurate answer                                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝


The knowledge graph lives in two moments:

At build time it is constructed alongside the vector store — both are populated from the same 
source documents, just processed differently. 
The vector store chunks and embeds text. 
The KG extracts named entities and the relationships between them.

At query time it is queried in parallel with the vector store. 
The vector store handles semantic similarity ("find me passages that feel like this question"). 
The KG handles relational traversal ("follow the edges from these specific entities and collect structured facts").

Both outputs land in the context builder, which assembles them into a single prompt for the LLM. 
The LLM never touches either store directly — it only sees the merged context window that was prepared for it.

## Exploration 2

Q)  So is ontology defined before knowledge graph initiation like can we set rules 
    for what kind of entities we are looking to establish

Yes, exactly — and this is one of the most important design decisions when building a knowledge graph system.

## **Ontology-first vs. data-first**

There are actually two schools of thought on the order of operations:

Ontology-first (top-down) — You define the schema before ingesting a single document. 
                            You decide upfront: what entity types exist, what relationships are valid between them, 
                            what properties each entity must have, and what constraints apply. 
                            The graph is then built to conform to that schema. 
                            This is how enterprise and domain-specific knowledge graphs are built — medical 
                            ontologies like SNOMED CT, legal graphs, financial compliance systems. 
                            The rules come first, the data fits into them.
                            
Data-first (bottom-up) — You ingest data, let entities and relationships emerge inductively, 
                         and refine the ontology iteratively as patterns appear. 
                         This is more common in research and exploratory settings where you don't 
                         know the shape of your data upfront.
                         
Most production systems use a hybrid: a core ontology defined upfront, then extended as new entity types are discovered.
              
What an ontology actually specifies

Think of it as a typed schema with rules. It defines four things:

Classes — the allowed entity types     

        Person, Organization, Drug, Disease, Gene, ClinicalTrial
        
Properties — what attributes each class can have    

    Person:  name (string), dob (date), nationality (string)
    Drug:    name (string), dosage (float), half_life (hours)
    Disease: name (string), icd_code (string), chronic (bool)
    
Relations — what connections are valid between which class pairs

    (Person)  –[treats]→       (Disease)     ✓ allowed
    (Drug)    –[treats]→       (Disease)     ✓ allowed
    (Disease) –[treats]→       (Person)      ✗ blocked
    (Person)  –[discovered_by]→(Gene)        ✗ wrong direction
    
Constraints — cardinality and logical rules
    
    A Person can have ONE date of birth           (1:1)
    A Drug can treat MANY diseases                (1:N)
    If A –[parent_of]→ B, then B –[child_of]→ A  (inverse constraint)
    A Drug cannot –[treats]→ itself               (irreflexive)


## **How this plugs into a RAG pipeline**


                    ONTOLOGY
                  (defined first)
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Allowed       Valid         Property
      classes     relations      schemas
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
             Entity extraction NLP
             (only extracts types
              the ontology allows)
                       │
                       ▼
               Knowledge graph
             (conforms to schema)

When entity extraction runs over your documents, it uses the ontology as a filter and validator. 
If your ontology says only Person, Drug, and Disease are valid entity types, the extractor won't 
create a node for a city or a date — those get ignored or stored elsewhere. 
If a relation isn't permitted between two classes, it gets rejected before it ever enters the graph.


**A concrete example**

Say you're building a clinical knowledge graph. You define this ontology upfront:

    Classes:
        Doctor, Patient, Drug, Disease, Hospital
        
    Relations:
        Doctor   –[works_at]→   Hospital
        Doctor   –[prescribes]→ Drug
        Drug     –[treats]→     Disease
        Patient  –[has]→        Disease
        Patient  –[prescribed]→ Drug
        
    Constraints:
        Drug –[treats]→ Drug     ✗  blocked
        Patient –[works_at]→ Hospital  ✗  blocked
        Doctor must have: name, specialty  (required fields)


Now when the NLP extractor reads "Dr. Chen prescribed Metformin at City Hospital", it knows:

    Dr. Chen → Doctor ✓
    Metformin → Drug ✓
    City Hospital → Hospital ✓
    prescribes(Doctor → Drug) ✓ valid relation
    works_at(Doctor → Hospital) ✓ valid relation
    
But if it reads "the pharmacy dispensed aspirin to the clinic", it would reject pharmacy as an entity 
type because it's not in the ontology, and reject dispensed as a relation because 
Organization → Drug isn't a permitted edge.


**The practical tradeoff**

A tight ontology gives you a clean, queryable, consistent graph but you'll miss things you didn't anticipate. 
A loose ontology lets the data speak but you end up with a messy graph that's harder to reason over. 
The general recommendation is to start with a minimal but precise ontology covering the core entity types 
and relationships you know you need, then extend it deliberately as your data reveals gaps rather than 
trying to define everything upfront.

    
### Exploration 2:

"If a user omits certain rules when defining an ontology, does the system automatically infer and 
categorize the missing cases, or does it strictly adhere to the explicitly defined rules and 
disregard anything not covered ?"

The answer depends entirely on which layer of the system you're asking about. 
There's no single universal behavior. 

It splits into three distinct strategies:

The three approaches: 

Strict / closed-world — The system only accepts what the ontology explicitly permits. 
                        Anything not defined is rejected or discarded. 
                        This is how formal ontology systems like OWL (Web Ontology Language) work by default. 
                        If Pharmacy isn't a class in your schema, that entity simply doesn't enter the graph. 
                        Nothing is inferred, nothing is auto-created.
                  
                       
Open / best-fit — The system tries to map unknown entities to the closest existing class.
                  If it sees Pharmacy and your ontology has Organization, it maps it there rather than dropping it. 
                  Most NLP-based extraction pipelines do this — they have a fallback category like Entity or 
                  Unknown that catches anything unclassified.
                  
                  
Adaptive / schema-relaxed — The system flags unknown types as candidates for new ontology classes rather than forcing 
                            them into existing ones or dropping them. 
                            It accumulates these in a "pending review" bucket and lets a human or automated process 
                            decide whether to extend the ontology. 
                            This is how graph databases like Neo4j work in practice — they don't enforce a rigid 
                            schema at the database level, so new node labels can appear at any time.

                            
What actually happens in a real RAG pipeline: 

In practice, most production systems use a layered approach:

    Incoming entity from NLP extractor
                  │
                  ▼
        Does it match a defined class?
             │              │
            YES              NO
             │              │
             ▼              ▼
        Accept +       Is there a
        validate       fallback class?
                        │        │
                       YES        NO
                        │        │
                        ▼        ▼
                  Map to       Log it +
                  fallback     discard
                  (e.g.        (strict
                  "Entity")    mode)


The most common real-world setup is that the ontology defines core classes strictly and has one or two catch-all 
fallback classes like Entity or Concept for anything that doesn't fit. 
So you don't lose data, but you also don't pollute your well-defined classes with noise.

The practical consequences of forgetting a rule

Depending on which behavior your system uses, forgetting to define Pharmacy as a class has very different outcomes:


    +-----------------------+------------------------------------------------------+
    | System mode           | What happens to Pharmacy                             |
    +-----------------------+------------------------------------------------------+
    | Strict / closed       | Silently dropped. Gone from the graph.               |
    +-----------------------+------------------------------------------------------+
    | Best-fit fallback     | Stored as `Organization` — loses precision           |
    |                       | but survives                                         |
    +-----------------------+------------------------------------------------------+
    | Catch-all             | Stored as `Entity: Pharmacy` — preserved             |
    |                       | but untyped                                          |
    +-----------------------+------------------------------------------------------+
    | Schema-relaxed        | Stored as its own label `Pharmacy` — graph           |
    | (Neo4j style)         | grows organically                                    |
    +-----------------------+------------------------------------------------------+
    

The silent drop in strict mode is the most dangerous because you don't know what you lost. 
Well-designed systems always log rejected entities somewhere, even if they don't enter the graph, 
so you can audit what the ontology is missing.

## **The recommended pattern**

Most mature systems handle this with a validation queue — a staging area between extraction and the graph:


    NLP Extractor
         │
         ▼
    Staging / validation layer
         │
         ├── Known class?  ──► Accept into graph
         │
         ├── Near match?   ──► Accept with flag for review
         │
         └── No match?     ──► Hold in queue + alert ontology owner
                                   │
                                   ▼
                            Human reviews queue,
                            extends ontology if needed,
                            re-processes held entities
                            

This way nothing is silently lost, the ontology evolves deliberately rather than chaotically, and the graph stays clean. 
The queue essentially becomes your early-warning system for ontology gaps — if the same unknown entity type keeps appearing, 
that's a strong signal it belongs in your schema.

"""

# ── Visual HTML ─────────────────────────────────────────────────────────────
# Return a full HTML string. It will be rendered in an iframe.
# Import from Required_Images/ or write inline.
VISUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { background: #0d0d0d; color: #e0e0e0; font-family: monospace;
         display: flex; justify-content: center; align-items: center;
         height: 100vh; margin: 0; }
  .placeholder { text-align: center; opacity: 0.5; }
  .placeholder h2 { font-size: 2rem; }
</style>
</head>
<body>
<div class="placeholder">
  <h2>📐 Visual Breakdown</h2>
  <p>HTML diagram coming soon for this topic.</p>
</div>
</body>
</html>
"""

# ── Step-by-Step Operations ─────────────────────────────────────────────────
OPERATIONS = {
    "Step 1: Example": {
        "description": "Brief description of what this step demonstrates.",
        "language": "python",
        "code": """
# Example code — runnable in subprocess
print("Hello from Step 1!")
""".strip(),
    },
    "Step 2: Example": {
        "description": "Next step description.",
        "language": "python",
        "code": """
print("Hello from Step 2!")
""".strip(),
    },
}

# ── Entry point called by topics/__init__.py ────────────────────────────────
def get_topic_data() -> dict:
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  VISUAL_HTML,
        "operations":   OPERATIONS,
    }
