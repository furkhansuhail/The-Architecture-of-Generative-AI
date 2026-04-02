"""
Sampling Theory
===============

How to draw conclusions about a population from a subset of it —
the mathematical foundation beneath every statistic, ML metric,
A/B test, and confidence interval you will ever compute.

"""

import textwrap
import re

TOPIC_NAME = "Google BigQuery"
DISPLAY_NAME = "04 . BigQuery"
ICON = "🎲"
SUBTITLE = "From Populations to Estimates — the Math Behind Every Statistic"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

What is Google BigQuery?
BigQuery is Google Cloud's fully managed, serverless data warehouse — a centralized system built to store, organize, 
and analyze massive amounts of data at incredible speed, without you needing to manage any servers or infrastructure.

Think of it as a giant, intelligent storage room where all your data lives in one place, and you can ask it any question 
using SQL and get answers within seconds — even if you're searching through billions of rows.


🏛️ A Data Warehouse to Centralize All Marketing Data
Instead of your data being scattered across different tools and platforms, BigQuery pulls everything into one central 
location. Whether it's website data, ad spend, or customer purchase history — it all lives together, making analysis 
much easier and more consistent.

🔗 Combine Advertising, CRM, Sales, and Analytics Data
This is one of BigQuery's biggest strengths. You can connect and merge data from completely different sources, for example:

Google Ads (how much you spent on ads)
CRM systems like Salesforce or HubSpot (customer info and sales pipeline)
Sales data (actual revenue and transactions)
Google Analytics (website traffic and behavior)

Without BigQuery, comparing these sources would require manually downloading reports and stitching them together 
in Excel — a slow and error-prone process.

🎯 Segment Audiences, Analyze Performance, Trigger Automations
Once all your data is centralized, you can:

Segment audiences — group customers by behavior, location, spending habits, etc.
Analyze performance — see which campaigns, products, or channels are actually driving results
Trigger automations — connect BigQuery to tools like Google Ads or Pub/Sub to automatically take action based on your 
data (e.g., automatically update ad audiences when users meet a certain condition)

✅ The Single Source of Truth
This is the ultimate goal. When everyone in a company — marketing, sales, finance, leadership — is pulling numbers 
from different tools, you often get conflicting reports. BigQuery solves this by being the one place where all data is clean, 
consistent, and agreed upon. Everyone works from the same numbers.

    
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │                    🔷  BigQuery = A Data Warehouse (Literally)                          │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────┐         ┌───────────────┐       ┌──────────────────────────────┐
    │   BigQuery = Warehouse       │         │               │       │      Query Results           │
    │  Your data storage facility  │         │               │       │  What you can do with data   │
    │                              │         │   SQL Query   │       │                              │
    │  ┌───────────┐ ┌───────────┐ │         │               │       │  Segment Audiences           │
    │  │           │ │           │ │         │  The warehouse│       │     Build custom cohorts     │
    │  │Advertising│ │   CRM     │ │         │     worker    │       │     for targeting            │
    │  │Meta, Goog,│ │HubSpot,   │ │         └───────────────┘       │                              │
    │  │  TikTok   │ │Salesforce │ │                                 │  Analyze Performance         │
    │  └───────────┘ └───────────┘ │    ┌──────────────────────┐     │     ROAS, CAC, LTV           │
    │  ┌───────────┐ ┌───────────┐ │    │ SELECT customer,     │     │     across channels          │
    │  │           │ │           │ │    │        revenue       │     │                              │
    │  │   Sales   │ │ Analytics │ │    │ FROM sales           │     │  ⚡ Trigger Automations       │
    │  │ Shopify,  │ │  GA4,     │ │    │ WHERE date > '2024-01│     │     Alerts, exports,         │
    │  │  Stripe   │ │ Mixpanel  │ │    └──────────────────────┘     │     workflows                │
    │  └───────────┘ └───────────┘ │              │                  │                              │
    │                              │              │                  │                              │
    │  Shelves = Tables            │              ▼                  │                              │
    │  Products = Rows of data     │           [ ──► ]               │                              │
    └──────────────────────────────┘                                 └──────────────────────────────┘
                  │                                                               ▲
                  │                  Data flows through SQL                       │
                  └───────────────────────────────────────────────────────────────┘
                  
                  
    What is a Marketing Data Warehouse?
    A central database where all your marketing data lives

    
    +----------------------+                +----------------------+
    | Meta                 |                |    HubSpot           |
    | Spend & Performance  |-----+    +-----|    Leads & Deals     |
    +----------------------+     |    |     +----------------------+
                                 |    |
    +----------------------+     v    v      +---------------------+
    | Google Ads           |  +-----------+  |   Shopify           |
    | Spend & Performance  |->|    DATA   |<-|  Orders & Revenue   |
    +----------------------+  | WAREHOUSE |  +---------------------+
                              |           |
    +----------------------+  +-----------+   +--------------------+
    | Google Analytics     |    ^    ^        |    Mailchimp       |
    |Sessions & Conversions|----+    +--------|    Email Campaigns |
    +----------------------+                  +--------------------+
    
    
            All connected  *  All queryable  *  All in one place


### BigQuery is Serverless

This means you don't manage any infrastructure to use it. Here's what that means in practice:
Traditional databases (NOT serverless):

You  -->  Rent a server  -->  Install database  -->  Configure it  -->  Query data
           (pay 24/7)         (your problem)        (your problem)

BigQuery (Serverless):

You  -->  Upload data  -->  Query data  -->  Done
          (Google handles EVERYTHING else)


+-------------------+---------------------+------------------------+
| Thing             | Traditional DB      | BigQuery               |
+-------------------+---------------------+------------------------+
| Servers           | You manage          | Google manages         |
| Scaling           | You configure       | Automatic              |
| Uptime            | You maintain        | Google's problem       |
| Storage limits    | You set             | Essentially unlimited  |
| Idle cost         | Paying even when    | Pay only when querying |
|                   | not used            |                        |
+-------------------+---------------------+------------------------+


Charges: 

    1) Data Storage 
    2) Streaming Inserts 
    3) Querying Data
    

The key practical benefits:

    * No setup — no installation, no configuration
    * Instant scale — can process terabytes in seconds without you doing anything
    * Pay-per-query — you're only billed for the data your SQL query actually scans
    * No idle cost — not running a query? You're not paying for a running server


Simple Analogy: It's like the difference between owning a car (traditional DB — insurance, maintenance, parking) vs. using 
Uber (BigQuery — you just say where you want to go and pay for the ride).



What is a Marketing Data Pipeline?
   Automated flow that extracts, transforms, and loads your marketing data


    +------------------+       +------------------+       +------------------+       +------------------+
    |                  |       |                  |       |                  |       |                  |
    |   Data Sources   |  -->  |    Transform     |  -->  |  Data Warehouse  |  -->  |    Analysis      |
    |                  |       |                  |       |                  |       |                  |
    | Meta, Google Ads,|       |  Clean, Blend,   |       |    BigQuery      |       | Reports and      |
    |   GA4, Shopify   |       |   Calculate      |       |                  |       | Dashboards       |
    |                  |       |                  |       |                  |       |                  |
    +------------------+       +------------------+       +------------------+       +------------------+



### What is an ETL Tool ?

ETL
    (Extrant -> Transform -> Load)
    
               The methodology to build data pipelines

    +----------------------+       +----------------------+       +----------------------+
    |          E           |       |          T           |       |          L           |
    |                      |       |                      |       |                      |
    |       Extract        |  -->  |      Transform       |  -->  |        Load          |
    |                      |       |                      |       |                      |
    | Pull raw data        |       | Clean, combine,      |       | Send to              |
    | from APIs            |       | calculate            |       | destinations         |
    |                      |       |                      |       |                      |
    | Sources:             |       | SQL Operations:      |       | Destinations:        |
    | - Google Ads         |       |  +-------+-------+   |       | - BigQuery           |
    | - TikTok             |       |  | JOIN  | WHERE |   |       | - Looker             |
    | - Shopify            |       |  +-------+-------+   |       | - Google Sheets      |
    | - HubSpot            |       |  | SUM   |CASE   |   |       |                      |
    |                      |       |  +-------+-------+   |       |                      |
    |                      |       |  |GROUP BY       |   |       |                      |
    |                      |       |  +---------------+   |       |                      |
    +----------------------+       +----------------------+       +----------------------+


Breaking down each step:
    
    E — Extract
    
        * Connects to your platforms via APIs
        * Pulls raw, unprocessed data as-is
        * Example: pulling all Meta ad spend data for the last 30 days
    
    T — Transform
    
    This is where SQL does the heavy lifting
        * JOIN — combine data from multiple sources (e.g. ad spend + revenue)
        * WHERE — filter only the rows you need
        * CASE — create conditional logic (e.g. if channel = 'meta' then 'Paid Social')
        * SUM / GROUP BY — aggregate data into summaries
    
    L — Load
    
    Pushes the clean, transformed data into its final destination
    Usually BigQuery, then connected to dashboards
    

ETL is the recipe. The data pipeline is the kitchen. BigQuery is the fridge.


ETL Tool: 

    What is an ETL Tool?

        +------------------------------------------------------------------+
        |                                                                  |
        |  Software that AUTOMATES the Extract, Transform, and Load        |
        |  process. It connects to your data sources, pulls the data,      |
        |  cleans it, and sends it to your data warehouse                  |
        |                  -- without writing code.                        |
        |                                                                  |
        +------------------------------------------------------------------+

                              POPULAR ETL TOOLS

        +----------+----------+----------+----------+----------+----------+
        |          |          |          |          |          |          |
        |  Porter  | Fivetran | Stitch   | Airbyte  |Supermet- |Funnel.io |
        |          |          |          |          |  rics    |          |
        +----------+----------+----------+----------+----------+----------+
        
        
Marketing ETL vs General ETL Tools

        +---------------------------+----------+---------------------------+
        |   ETL Tools for Marketing |    VS    |    General ETL Tools      |
        +---------------------------+----------+---------------------------+
        |                           |          |                           |
        | OAuth login, data flows   |  SETUP   | Define schemas, map       |
        | in 5 min                  |          | fields, write YAML        |
        |                           |          |                           |
        +---------------------------+----------+---------------------------+
        |                           |   DATA   |                           |
        | Ad platforms, GA4, CRM    | SOURCES  | Databases, APIs,          |
        |                           |          | SaaS apps                 |
        +---------------------------+----------+---------------------------+
        |                           | TRANSFOR-|                           |
        | Pre-built: ROAS, CPA,     | MATIONS  | Write custom              |
        | CTR, CAC, LTV             |          | SQL transforms            |
        |                           |          |                           |
        +---------------------------+----------+---------------------------+
        |                           |  TARGET  |                           |
        | Marketers & analysts      | AUDIENCE | Data engineers            |
        |                           |          |                           |
        +---------------------------+----------+---------------------------+
        |                           |  TIME TO |                           |
        | 5-15 minutes              |  VALUE   | Hours to days             |
        |                           |          |                           |
        +---------------------------+----------+---------------------------+
        |                           |          |                           |
        | Per connector/account     | PRICING  | Per row/volume            |
        |                           |          |                           |
        +---------------------------+----------+---------------------------+
        



Big Query Data Setup and breakdown 

    
        Datasets & Tables
             How your data is organized in BigQuery
    
    
    +--------------------------------------------------+
    |  Project                                         |
    |                                                  |
    |  +--------------------------------------------+  |
    |  |  Dataset                                   |  |
    |  |                                            |  |
    |  |  +------------+ +----------+ +----------+  |  |
    |  |  | Table1     | |Table2    | |Table3    |  |  |
    |  |  |            | |          | |          |  |  |
    |  |  | google_ads | | meta_ads | |ga4_events|  |  |
    |  |  |  (table)   | | (table)  | | (table)  |  |  |
    |  |  +------------+ +----------+ +----------+  |  |
    |  |                                            |  |
    |  +--------------------------------------------+  |
    |                                                  |
    +--------------------------------------------------+
    
            Full path: project.dataset.table




                    +------------------+
                    |   Organization   |  <-- Highest-level node in BigQuery
                    +------------------+
                             |
                          CONTAINS
                             |
          +-------------+--------------+--------------+--------------+
          |             |              |              |              |
      [Project A]   [Project B]    [Project C]    [Project D] ...[Project N]
                                       |
                                    LINKED TO
                                       |
                        +--------------+--------------+
                        |                             |
                   SCENARIO A                    SCENARIO B
                        |                             |
              One Billing Account               Multiple Billing Accounts
              covers all projects               each assigned to a subgroup
                          |                           |
             +---+---+---+---+---+            +---+---+---+         +---+---+
             | A | B | C | D | N |            | A | B | C |         | D | N |
             +---+---+---+---+---+            +---+---+---+         +---+---+
                       |                            |                   |
               +----------------+           +----------------+  +----------------+
               | Billing Acct 1 |           | Billing Acct 1 |  | Billing Acct 2 |
               +----------------+           +----------------+  +----------------+





## IAM (Identity and Access Management) in BigQuery:


The Access Control Hierarchy

    Organization
    |
    └── Project  <-- Access can be granted here
            |
            └── Dataset  <-- Access can be granted here
                    |
                    └── Table  <-- Access can be granted here (most granular)
                    
    The lower you grant access, the more specific and restricted it is.
    
    
    +------------------+------------------------------------------+
    | Role             | What they can do                         |
    +------------------+------------------------------------------+
    | Viewer           | Read data only, no edits                 |
    | Editor           | Read + write data                        |
    | Owner            | Full control including deleting          |
    | Data Viewer      | Query tables in a dataset                |
    | Data Editor      | Insert/update/delete rows                |
    | Job User         | Run queries (but not see all data)       |
    | Admin            | Manage everything in the project         |
    +------------------+------------------------------------------+
    
    
    
How Access Layers Work

    SCENARIO: Marketing Agency managing client data

    Project: Client_XYZ
        |
        +-- Dataset: meta_ads      --> Marketing team (Viewer)
        |
        +-- Dataset: finance_data  --> Finance team only (Viewer)
        |                              Marketing team has NO access
        |
        +-- Dataset: raw_data      --> Data engineers only (Editor)


Who can be granted access ?
    
    +----------------------+--------------------------------+
    | Identity Type        | Example                        |
    +----------------------+--------------------------------+
    | Google Account       | john@gmail.com                 |
    | Google Group         | marketing-team@company.com     |
    | Service Account      | etl-pipeline@project.gsiam...  |
    | Domain-wide          | Everyone @yourcompany.com      |
    +----------------------+--------------------------------+

Service accounts are especially important — they're how ETL tools like Fivetran or your data pipelines get authorized 
access to BigQuery without using a personal login.





Practical Example

    Data Engineer  -->  Project Owner   (can do everything)
    Analyst        -->  Data Viewer     (can query, cannot delete)
    ETL Pipeline   -->  Data Editor     (can write data in, cannot delete tables)
    Executive      -->  Viewer          (can see dashboards, cannot touch raw data)
    
Bottom line: IAM lets you apply the principle of least privilege — every person or system only gets exactly the access 
they need, nothing more.

    
    
    
    
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {


}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    """Return all content for this topic module — single source of truth."""
    return {
        "display_name": DISPLAY_NAME,
        "icon": ICON,
        "subtitle": SUBTITLE,
        "theory": THEORY,
        "visual_html": "",
        "visual_height": 400,
        "complexity": None,
        "operations": OPERATIONS,
    }