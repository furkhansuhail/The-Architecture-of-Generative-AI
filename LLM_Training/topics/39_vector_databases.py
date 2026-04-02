"""
Kubernetes Fundamentals
========================
From "what is K8s" to Pods, Deployments, Services, ConfigMaps,
and the core workflow of deploying containerized applications at scale.
"""
import re

TOPIC_NAME = "39 . Vector Databases"
DISPLAY_NAME = "39 · Vector Databases"
ICON         = "🧠"
SUBTITLE     = "Vector Databases"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### Vector Databases

The core idea is this: instead of searching data by exact keyword matches (like a traditional database), 
a vector database searches by meaning. 

It does this by converting everything — text, images, audio — into lists of numbers called embeddings, and then 
finding items whose numbers are geometrically close in space.



Here's a breakdown of the key ideas:
Embeddings are the magic ingredient. An embedding model (like OpenAI's text-embedding-ada or sentence-transformers) 
reads a piece of content and outputs a long list of numbers — typically 768 to 3,000+ dimensions — that captures its 
semantic meaning. 
Two pieces of content that mean similar things will produce vectors that are close together in this high-dimensional space.


Similarity = distance. When you search, your query gets embedded too, and the database finds the stored vectors that are 
geometrically nearest (using cosine similarity or Euclidean distance). This is why a search for "puppy training" can 
return results tagged "dog obedience" — they land near each other in vector space even without sharing any keywords.


Fast search at scale. Searching through millions of vectors brute-force would be too slow, so vector databases use 
indexing algorithms like HNSW (Hierarchical Navigable Small World graphs) or IVF (Inverted File Index) to find approximate 
nearest neighbors in milliseconds. 
Popular vector databases include Pinecone, Weaviate, Qdrant, Chroma, and pgvector (a Postgres extension).


Where they're used. The most common application right now is RAG (Retrieval-Augmented Generation): you embed your documents, 
store them in a vector DB, and when a user asks a question, you retrieve the most relevant chunks and pass them to an 
LLM as context. This is how products like Notion AI, customer support bots, and document Q&A tools work under the hood.



### Indexing algorithms

There are two dominant approaches: IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World). 
They solve the same problem — "don't compare against every vector" — but in completely different ways. 



1. IVF (Inverted File Index)

    IVF works by pre-partitioning the space with k-means clustering. During indexing, every vector is assigned to its 
    nearest centroid. At query time, you only compute distances to the centroids (cheap — just a handful), then search 
    within the nearest cluster. The tradeoff: you can miss results that live just across a cluster boundary. 
    That's why production systems often use nprobe — 
    

2. HNSW (Hierarchical Navigable Small World graphs)
    
    HNSW takes a completely different approach — it builds a multi-layer graph where each layer is a progressively 
    finer view of the same data. The search descends from coarse to fine, like zooming in on a map.

    HNSW works like a highway system. The top layer is the interstate — few nodes, huge jumps. The bottom layer is city 
    streets — every node, fine-grained connections. You start on the interstate, drive toward the destination, then exit 
    onto progressively local roads. Because you skip entire regions at the high layers, you arrive near the target very 
    quickly before doing the precise local search at Layer 0.
    
    The probability that a node appears in a higher layer is set by a parameter (typically ~1/ln(M), where M is the 
    number of connections per node). 
    Nodes promoted to higher layers act as "hubs" that let the algorithm teleport across the space.
    
    Here's how the two algorithms compare in practice:
        IVF is simpler, uses less memory, and works well when your dataset is relatively static. 
        The main risk is boundary misses — a result that lives just outside the searched cluster gets dropped. 
        You tune nprobe (number of clusters to search) to trade speed vs. recall.
    
HNSW has better recall at the same speed, handles dynamic inserts well (you can add new vectors without rebuilding), 
and doesn't require training a clustering model first. The cost is higher memory usage — every node stores its list of 
neighbors at each layer.

Most production vector databases (Qdrant, Weaviate, Milvus) default to HNSW for this reason. 
Some offer hybrid indexes that combine IVF coarse partitioning with HNSW fine navigation for the best of both worlds 
at billion-vector scale.


### How are vectors stored, and does querying them not become too slow as the database grows?

Vector databases follow a structure conceptually similar to a relational database table. Each row stores the original 
document or text chunk, its corresponding embedding vector — a list of typically 768 to 3,072 floating point numbers — 
and any associated metadata such as source, date, or user identifiers. The key difference from a traditional database 
is that retrieval is not based on exact keyword matching but on geometric proximity: finding the rows whose vectors are 
closest to the query vector in high-dimensional space.

A naïve implementation would scan every row on each query, comparing the query vector against all stored vectors one by one. 
At scale, with millions of documents and high-dimensional embeddings, this brute-force approach would be computationally 
prohibitive. Vector databases solve this by building an index structure alongside the raw data at insert time. 
The most widely used is HNSW (Hierarchical Navigable Small World), a layered graph where upper layers enable fast 
long-range jumps across the vector space and lower layers refine the search locally. At query time, the system navigates 
this graph rather than scanning the data, reducing search complexity from O(n) to approximately O(log n). ChromaDB, for 
instance, stores the original documents and metadata in a SQLite file and maintains a separate hnswlib file containing 
the pre-built HNSW graph. The two are kept in sync automatically, so the index is always ready without any manual intervention.




### How Are Vectors Stored in a Vector Database, and Does Querying Them Not Become Prohibitively Slow as the Database Grows?

To understand how vector databases store and retrieve data efficiently, it is first necessary to understand what a vector 
actually is and why storing text as a vector is useful in the first place. When a piece of text — whether a single word, 
a sentence, a paragraph, or an entire document — is passed through an embedding model, it is transformed into a 
fixed-length sequence of floating point numbers. This sequence is called an embedding vector, or simply a vector. 
Depending on the model used, this sequence might contain 384 numbers (for lightweight models like all-MiniLM-L6-v2), 
1,536 numbers (for OpenAI's text-embedding-3-small), or as many as 3,072 numbers (for text-embedding-3-large). 
Each number in this sequence encodes some aspect of the meaning, context, and semantic content of the original text, 
and the entire sequence together constitutes a coordinate in an extremely high-dimensional geometric space.


The critical property that makes this useful is that semantically similar texts produce vectors that are geometrically 
close to one another. "A dog ran through the park" and "A puppy sprinted across the field" share no significant keywords, 
yet an embedding model will place their vectors very near each other in this high-dimensional space because both 
sentences describe the same kind of event. Conversely, "quantum mechanics" and "Italian pasta recipes" will be placed 
far apart. This geometric encoding of meaning is what allows vector search to retrieve semantically relevant results 
even when the query and the document share no exact words — a capability that traditional keyword-based systems like 
SQL full-text search cannot replicate.

## The Storage Structure of a Vector Database

At its core, a vector database stores three things for each entry: the original content (the raw text, image, or other 
data that was embedded), the embedding vector itself, and any associated metadata. The metadata can include virtually 
any structured information relevant to the application — the source document, a timestamp, a user identifier, a category 
label, a URL, or any other field that might be useful for filtering results. If one were to represent this as a table 
analogous to a relational database, a single row might look as follows: an identifier column, a document column 
containing the original text, an embedding column containing a list of 1,536 floating point numbers, and a metadata 
column containing a JSON object with auxiliary information.

This structure is more flexible than a traditional relational database in some ways and more constrained in others. 
Unlike MySQL or PostgreSQL, vector databases do not require a fixed schema where every row has the same set of columns 
with predefined data types. Different rows in the same collection can carry different metadata fields. However, the
embedding column is always of a fixed dimensionality — every vector in a given collection must have the same number of 
dimensions, because comparisons between vectors only make sense when they live in the same geometric space. 
Mixing 1,536-dimensional vectors from one model with 384-dimensional vectors from another in the same collection would 
be meaningless, in the same way that comparing a two-dimensional coordinate to a three-dimensional one is undefined.

In ChromaDB specifically, which is one of the most accessible vector databases and is commonly used in local and prototype 
applications, this data is stored across two separate files on disk. The first is a SQLite database file, which holds 
the document text, the raw embedding vectors in serialised form, and all associated metadata. SQLite is a lightweight, 
file-based relational database engine that requires no server process, making it appropriate for local use. 
The second file is maintained by a library called hnswlib, which is a highly optimised C++ implementation of the HNSW algorithm. 
This file does not store the original documents or vectors in a human-readable format — rather, it stores a graph data 
structure that encodes the neighbourhood relationships between vectors in the collection. These two files serve entirely 
different purposes: the SQLite file is the source of truth for the actual data, and the hnswlib file is the index that 
makes fast search possible. When a new document is added to a ChromaDB collection, both files are updated — the document 
and its vector are written to SQLite, and the vector is inserted into the HNSW graph. When a query is executed, 
ChromaDB uses the HNSW graph to find the nearest neighbours efficiently, then retrieves the corresponding documents and 
metadata from SQLite using their identifiers.

## Why Naïve Search Would Be Too Slow

The concern about querying being slow is entirely well-founded, and it strikes at the heart of why vector databases 
exist as a distinct category of software rather than simply being a feature added to existing relational databases. 
To appreciate the problem, consider what a search without any index structure would look like.

A query arrives as a piece of text — say, "what are the health benefits of green tea?" This text is passed through the 
same embedding model that was used to embed the stored documents, producing a query vector of 1,536 floating point 
numbers. The task is now to find the k stored vectors that are most similar to this query vector, where similarity is 
typically measured using cosine similarity or Euclidean distance. Cosine similarity measures the angle between two 
vectors, with a value of 1 indicating they point in exactly the same direction (maximum similarity) and 0 indicating 
they are perpendicular (no similarity). Euclidean distance measures the straight-line distance between two points in 
the high-dimensional space, with smaller values indicating greater similarity.

Without any index, the only way to find the most similar vectors is to compute the distance or similarity between the 
query vector and every single stored vector, one by one, and then sort the results to find the top k. 
This is called brute-force search or exact nearest neighbour search. For a collection with one thousand documents, 
this is perfectly tractable — one thousand distance computations take milliseconds. But vector databases are not built 
for thousands of documents; they are built for millions, tens of millions, or billions. At one million documents, 
each with a 1,536-dimensional vector, a single query requires one million distance computations, 
each involving 1,536 multiplications and additions. The total number of arithmetic operations per query is on the order 
of one billion. Even on modern hardware with vectorised floating point units, this begins to take hundreds of milliseconds 
to seconds per query, which is far too slow for any real-time application. At ten million documents the problem is ten 
times worse, and it scales linearly — doubling the number of documents doubles the query time. 
This O(n) scaling, where n is the number of stored vectors, makes brute-force search fundamentally unsuitable for 
large-scale production use.

There is also a storage dimension to this problem. A single 1,536-dimensional float32 vector occupies 1,536 times 4 bytes, 
which is 6,144 bytes, or approximately 6 kilobytes. One million such vectors occupy roughly 6 gigabytes of storage 
purely for the vectors themselves, before accounting for the original documents and metadata. Loading 6 gigabytes of 
data from disk into memory for every query is clearly impractical. Even if the vectors are kept in memory (which is the 
approach taken by many production vector databases), computing distances against all of them for every query is still 
the bottleneck. The challenge is therefore both computational and architectural: how can the nearest neighbours be 
found without examining all stored vectors?

## Approximate Nearest Neighbour Search and Why Exactness Can Be Traded

The answer lies in a conceptual trade-off that initially seems surprising: in most practical applications, finding the 
exact nearest neighbour is not necessary. Finding a very good approximate nearest neighbour — one that is within a small 
margin of the true nearest — is sufficient, and this approximation can be achieved orders of magnitude faster than exact 
search. This is the foundation of Approximate Nearest Neighbour (ANN) search, and it is the algorithmic basis for all 
high-performance vector databases.

To understand why approximation is acceptable, consider the use case. If a user asks a question-answering system 
"what are the health benefits of green tea?" and the system retrieves the third, fourth, and fifth most semantically 
similar documents rather than the first, second, and third, the quality of the answer will in practice be 
indistinguishable to the user. The documents that are semantically very close to the query are all highly relevant, 
and there is no meaningful difference between them from the user's perspective. What matters is that the retrieval is 
fast and that the results are good, not that they are provably optimal. Recall — the fraction of true nearest neighbours 
that are included in the returned results — is the metric that quantifies this trade-off, and a recall of 95% at ten 
times the speed of exact search is almost universally preferable to 100% recall at baseline speed.

The IVF Index: Partitioning Space into Clusters
One of the earliest and most widely used ANN index structures is the Inverted File Index, typically abbreviated as IVF. 
The core idea is conceptually straightforward: rather than searching all stored vectors, partition the vector space into 
a number of clusters and search only the cluster or clusters most likely to contain the nearest neighbours.

During the indexing phase, which happens offline before any queries are served, 
a clustering algorithm — specifically k-means — is run over all stored vectors. K-means partitions the vectors into a 
predefined number of clusters, typically denoted nlist, where each cluster is represented by a centroid vector. 
The centroid is the average position of all vectors assigned to that cluster. Once the clusters are formed, each stored 
vector is assigned to its nearest centroid and stored in the corresponding inverted list — a list of all vectors 
belonging to that cluster. This is the origin of the name Inverted File Index: like an inverted index in text search, 
it maps from a key (the centroid) to a list of entries (the vectors nearest to that centroid).

At query time, the query vector is first compared against all nlist centroids. 
This is a small number of comparisons — perhaps 256 or 1,000, regardless of how many millions of vectors are stored. 
The cluster or clusters whose centroids are nearest to the query vector are identified, and then the full distance 
computation is performed only against the vectors within those clusters. 
If there are one million vectors and 256 clusters, each cluster contains on average roughly 4,000 vectors. 
Searching a single cluster therefore requires about 4,000 distance computations instead of one million — a 250-fold 
reduction in work.

The number of clusters searched is controlled by a parameter called nprobe. Setting nprobe to 1 searches only the 
single nearest cluster, which is fastest but risks missing relevant vectors that sit just across a cluster boundary. 
Increasing nprobe to 4 or 16 searches multiple clusters, improving recall at the cost of some additional computation. 
The relationship between nprobe, recall, and query speed is a smooth curve that can be tuned to match the requirements 
of a given application. A system that prioritises speed and can tolerate 85% recall might use nprobe of 4, while a 
system requiring 99% recall would set nprobe to 64 or higher.

The fundamental limitation of IVF is the cluster boundary problem. Vector space does not divide cleanly into 
non-overlapping regions, and a query vector that falls near the boundary between two clusters may have its true nearest 
neighbours in the adjacent cluster that was not searched. This is not a rare edge case — in high-dimensional spaces, 
a significant fraction of queries land near cluster boundaries because the concept of a boundary becomes increasingly 
diffuse as dimensionality increases. IVF handles this through nprobe, but there is a fundamental tension between 
minimising the number of clusters searched and maintaining high recall.

The HNSW Index: A Hierarchy of Navigable Graphs
The Hierarchical Navigable Small World (HNSW) algorithm, introduced by Malkov and Yashunin in 2018, takes a 
fundamentally different approach to the nearest neighbour problem. Rather than partitioning space into clusters, 
HNSW builds a multi-layered graph structure where each node represents a stored vector and edges connect vectors that 
are nearby in the high-dimensional space. The key innovation is the hierarchical structure: the graph is organised into 
multiple layers, with each successive layer containing progressively more nodes and progressively shorter-range connections.

The analogy to a road network is apt and instructive. A highway system has a small number of on-ramps and very long 
stretches of road that cover large distances quickly. A city road network has far more intersections and shorter road 
segments that allow precise navigation within a small area. A neighbourhood street network is the most detailed, 
with the greatest density of connections and the finest granularity of navigation. HNSW builds exactly this kind of 
layered network over the vector space.

In the top layer — Layer 2 in a typical three-layer configuration — only a small subset of nodes are present, perhaps 
three to five percent of the total, and they are connected to each other with long-range edges that span large portions 
of the vector space. In Layer 1, a larger subset of nodes is present with medium-range connections. In Layer 0, the base 
layer, all nodes are present and connected to their immediate neighbours in vector space. The probability that a given 
node is promoted to a higher layer is determined stochastically during index construction, following a logarithmic 
distribution that ensures the upper layers remain sparse.

Search proceeds by navigating this hierarchy in a top-down greedy manner. The query begins at a fixed entry point in 
the topmost layer. At each step, the algorithm examines the neighbours of the current node in the current layer and 
moves to whichever neighbour is closest to the query vector. This greedy walk continues until no neighbour in the 
current layer is closer to the query than the current node — that is, until a local optimum is reached. 
The algorithm then drops down to the next layer, resuming the greedy walk from the position where it stopped. 
This process repeats until the bottom layer (Layer 0) is reached, at which point the final greedy walk through the 
dense local neighbourhood produces the approximate nearest neighbours.

The reason this works so efficiently is that the upper layers serve as a coarse navigation mechanism, allowing the 
algorithm to quickly skip irrelevant portions of the vector space in large jumps. By the time the search reaches Layer 0, 
it is already in the correct neighbourhood of the query vector and only needs to explore a local patch of nodes. 
The total number of distance computations is proportional to the depth of the hierarchy, which grows logarithmically 
with the number of stored vectors. 

This gives HNSW its characteristic O(log n) query complexity — doubling the number of stored vectors adds only a single 
step to the search, rather than doubling the work as brute-force search would.

HNSW is controlled by three main parameters. The parameter M defines the number of bidirectional connections each 
node maintains in the graph. Higher values of M produce a better-connected graph with higher recall but require more 
memory and slower index construction. Typical values range from 16 to 32. The parameter ef_construction controls the 
beam width used during index construction — how many candidate neighbours are considered when inserting each new node 
into the graph. Higher values produce a higher-quality graph at the cost of slower insertion. 
The parameter ef_search, also written simply as ef at query time, controls the beam width during the search itself. 
It determines how many candidate nodes are kept in the priority queue as the algorithm navigates the graph. Higher 
ef_search values increase recall by exploring more of the graph but reduce query speed. Setting ef_search to 64 
typically yields around 97% recall, while ef_search of 128 approaches 99% recall.

A particularly important property of HNSW is that it supports dynamic insertions efficiently. New vectors can be 
added to the graph without rebuilding the entire index from scratch, which is critical for applications where the 
document collection grows continuously. IVF, by contrast, requires periodic re-clustering when new vectors are added, 
since the cluster assignments become stale as the distribution of vectors changes.

## Comparing IVF and HNSW in Practice

Both IVF and HNSW offer substantial speedups over brute-force search, but they have different trade-off profiles that 
make each more suitable for different scenarios. Benchmarks consistently show that HNSW achieves higher recall at the 
same query speed, or equivalently, the same recall at higher query speed. When plotted as recall versus queries per second, 
HNSW traces a curve that lies consistently above and to the right of IVF, meaning that for any given recall target, 
HNSW can serve more queries per second. This property is called Pareto dominance — HNSW is not merely better on one 
dimension but better across the entire trade-off frontier.

However, IVF has a significant advantage in memory consumption. The HNSW graph stores the neighbourhood connections 
for every node, and with M connections per node the memory overhead can be substantial. At M equals 16, 
each node requires 16 additional pointers per layer, and for a collection of ten million vectors this adds up to 
several gigabytes of graph metadata above and beyond the raw vectors. IVF, by contrast, requires only a list of centroid 
vectors and an assignment of each stored vector to its nearest centroid, which is far more compact. 
For applications running on memory-constrained hardware or handling very large collections where the entire index cannot 
fit in RAM, IVF may be the only practical choice.

A further consideration is that IVF requires a training phase. The k-means clustering must be run over a representative 
sample of the data before any vectors can be inserted, and the quality of the clustering affects recall significantly. 
If the initial training data is not representative of the eventual collection, the cluster boundaries will be poorly 
placed and recall will suffer. HNSW has no training phase — vectors can be inserted immediately, and the graph 
quality degrades gracefully as the collection grows.

## How ChromaDB Implements This in Practice
ChromaDB is designed to abstract away the complexity of index management behind a simple Python API. 
When a collection is created and documents are added using the add() method, ChromaDB automatically embeds the documents 
(if an embedding function is provided), stores the vectors and documents in its SQLite backend, and inserts the vectors 
into an hnswlib index. The hnswlib library is a highly optimised C++ implementation of HNSW that ChromaDB calls via 
Python bindings. All of this happens transparently — the developer specifies what to store and ChromaDB handles 
how it is stored and indexed.

The persistence model in ChromaDB is straightforward. By default, ChromaDB operates in-memory and all data is lost 
when the process exits. To enable persistence, a path is passed to the Client constructor, and ChromaDB will read and 
write its SQLite and hnswlib files to that directory. This is the "stored locally in a file" behaviour that many users 
first encounter. 
The on-disk format is simply two files in the specified directory: chroma.sqlite3, which contains the documents, 
vectors in serialised binary form, and metadata, and a directory containing the hnswlib index files.

When a query is executed using the query() method, ChromaDB embeds the query text, passes the query vector to 
hnswlib, retrieves the nearest neighbour identifiers from the graph, and then fetches the corresponding documents and 
metadata from SQLite using those identifiers. The two steps — graph traversal and document retrieval — are decoupled, 
which means the slow part (graph traversal) is handled by the optimised C++ HNSW implementation, and the fast part 
(document lookup by primary key) is handled by SQLite.

#### Quantisation: Compressing Vectors to Save Memory and Speed Up Search

Production vector databases frequently employ an additional technique called quantisation to reduce memory consumption 
and speed up distance computations at the cost of a small reduction in recall. The most common form is 
Product Quantisation (PQ), which works by dividing the high-dimensional vector into a number of sub-vectors and 
replacing each sub-vector with a representative code from a learned codebook. Instead of storing 1,536 float32 values 
per vector (6,144 bytes), a product-quantised representation might store 96 one-byte codes (96 bytes), a compression 
ratio of 64 times. Distance computations can then be performed using lookup tables in the compressed space rather than 
arithmetic in the original space, which is significantly faster.

The trade-off is that distances computed in the compressed space are approximate, introducing a small additional source 
of error beyond the approximation already inherent in ANN search. In practice, PQ compression with a reasonable number 
of sub-vectors introduces only a small recall penalty while dramatically reducing memory requirements and increasing 
throughput. Many production systems use a two-stage search: PQ-compressed vectors are used to identify a small candidate 
set quickly, and then the exact distances are recomputed using the original uncompressed vectors to re-rank the 
candidates and produce the final results. This approach, called asymmetric distance computation, combines the speed 
benefits of quantisation with the accuracy of exact distance computation for the final ranking step.

## The Broader Landscape: Filtering and Hybrid Search
One aspect of vector database querying that differs importantly from pure nearest neighbour search is the frequent 
need to combine vector similarity with metadata filters. A real application might need to find the ten documents most 
semantically similar to the query, but only among documents authored in the last thirty days, or only among documents 
belonging to a specific user. This is called filtered vector search, and it introduces significant complexity because 
the filter interacts with the index in non-trivial ways.

A naïve approach — post-filtering — retrieves the top k nearest neighbours ignoring the filter, then discards any that 
do not satisfy the filter. This can produce fewer than k results if many of the nearest neighbours are filtered out, 
and in pathological cases it can return zero results even though matching documents exist. 
The correct approach — pre-filtering — applies the filter before the vector search, restricting the search to only the 
subset of vectors that satisfy the filter. This requires that the vector index support filtered search efficiently, 
which not all implementations do. Qdrant, for example, stores payload metadata alongside vectors in a columnar format 
designed for efficient pre-filtering, and it applies filters during the HNSW graph traversal rather than after the fact. 
This ensures that the returned results always satisfy the filter and that the search explores only the relevant subset 
of the graph.

A related concept is hybrid search, which combines vector similarity with keyword relevance. 
A system using hybrid search returns results that are both semantically similar to the query and contain relevant 
keywords, combining the strengths of embedding-based retrieval and classical text search. 
Weaviate implements this natively by maintaining both an HNSW index for vector similarity and a BM25 inverted index 
for keyword relevance, and combining their scores using a parameter called alpha that interpolates between pure 
keyword search (alpha equals 0) and pure vector search (alpha equals 1). This hybrid approach tends to outperform 
either method alone, particularly for queries that contain important specific terms — product codes, person names, 
technical acronyms — that an embedding model might not distinguish sufficiently from near-synonyms.

### Summary
A vector database is therefore best understood not as a fundamentally different kind of system from a relational 
database, but as a system that adds a specialised indexing capability on top of familiar storage primitives. 
The data is stored in rows, each containing the original document, its vector representation, 
and associated metadata — much like a table in a relational database. The critical difference lies in the query 
mechanism and the index structure that supports it. Rather than B-tree indices optimised for exact equality and 
range queries, vector databases employ ANN index structures — primarily HNSW or IVF — that are optimised for 
proximity queries in high-dimensional space. These indices transform what would otherwise be an O(n) linear scan 
into an O(log n) graph traversal, making it possible to search collections of millions or tens of millions of vectors 
with query latencies measured in single-digit milliseconds. ChromaDB achieves this locally by maintaining a SQLite file 
for durable storage and an hnswlib file for fast indexed search, combining the two transparently so that developers 
interact only with a simple collection abstraction while the complexity of efficient approximate nearest neighbour 
search is handled automatically underneath.


"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = """




"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {


}



# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
# render_operations() has been removed.  app.py owns all Streamlit rendering
# via its own render_operation() helper and strips callables from topic dicts
# inside load_topics_for() anyway — so a local render function is never called.

def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────


def get_content():
    """Return all content for this topic module — single source of truth."""
    visual_html   = ""
    visual_height = 400
    try:
        from LLM_Training.visuals.vector_db_visual import (   # ← match your exact folder casing
            VECTORDB_VISUAL_HTML,
            VECTORDB_VISUAL_HEIGHT,
        )
        visual_html   = VECTORDB_VISUAL_HTML.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        visual_height = VECTORDB_VISUAL_HEIGHT
    except Exception as e:
        import warnings
        warnings.warn(f"Could not load visual: {e}", stacklevel=2)

    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   visual_html,
        "visual_height": visual_height,
        "complexity":    None, # COMPLEXITY,
        "operations":    None, # OPERATIONS,
    }