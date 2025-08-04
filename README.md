
# KNIT Universal Tool – Ontology Generator

KNIT Universal Tool is a flexible framework for **reusing and integrating ontologies** from multiple heterogeneous repositories.  
It automatically retrieves ontology fragments relevant to user-specified keywords, enriches them by adding taxonomic context, and merges them into a coherent OWL ontology with an **interactive visualisation**.

This repository contains the **Gradio-based interactive interface** of KNIT Universal, which connects to the **Storage Component API** (vectorisation + semantic search: https://github.com/rorevello/OntoVector-Search-API ) to locate and retrieve ontology classes across distributed repositories.


## 🚀 Features

- **Multi-repository ontology reuse** (BioPortal, OBO Foundry, W3ID, local).
- **Format-agnostic**: supports OBO, OWL, TTL, RDF, JSON-LD, etc.
- **Multiple search strategies**:
  - `knit`: frequency-based coverage.
  - `Nc`: total number of classes.
  - `DSRC`: semantic cohesion score.
  - `centroid`: proximity to the semantic centroid of the keywords.
- **Automatic enrichment**: retrieves ancestors, equivalent classes, and related properties.
- **Cleaning & optimisation**: removes empty property chains, blank nodes, and orphaned entities.
- **Interactive outputs**:
  - Downloadable OWL ontology.
  - Downloadable HTML visualisation.
  - Tabular list of RDF triples.

## 📦 Requirements


Install dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Usage

1. Start the Storage Component API (vector search backend).

2. Run the Gradio interface:

```bash
python main.py
```

3. Open the interface in your browser at the address shown in the terminal.
 
4. Provide:
 
- Keywords (one per line).
- Storage Component URL (e.g., http://0.0.0.0:8019).
- Ontology source (e.g., bioportal, obo_fundry, w3id, or all).
- Search strategy (knit, Nc, DSRC, centroid).
   
5. Click Submit and download:
- OWL file of the merged ontology.
- HTML graph visualisation.
- Table of triples.


