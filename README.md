# HubGenes

HubGenes is a gene-prioritization project built around **STRING protein-protein interaction data, graph topology, and machine learning**.

The current workflow:

* Builds a graph from STRING interactions
* Computes `betweenness`, `closeness`, `degree`, and `MCC` (Maximal Clique Centrality)
* Creates labels using a chosen rule
* Trains LASSO logistic regression, SVM-RFE, and Random Forest
* Ranks genes using a composite score

## Project Structure

* [`streamlit_app.py`](streamlit_app.py): Streamlit UI for the current median-label ranking workflow
* [`hubgenes/pipeline.py`](hubgenes/pipeline.py): Graph construction, topology feature generation, labeling, model training, evaluation, and ranking logic
* [`notebooks/ranking.ipynb`](notebooks/ranking.ipynb): Notebook using median-based labels across all four topology measures
* [`data/string_interactions_short.tsv`](data/string_interactions_short.tsv): STRING interaction dataset

## Methods

### Topological Measures

The pipeline computes:

* Betweenness centrality
* Closeness centrality
* Degree
* MCC (Maximal Clique Centrality)

### Labeling Strategies

The current labeling strategy uses the **median rule** in [`notebooks/ranking.ipynb`](notebooks/ranking.ipynb):

* Label = `1` if a gene is greater than or equal to the median for **all four** topology measures
* Otherwise, label = `0`

### Machine Learning Models

The ranking workflow uses:

* LASSO-style logistic regression
* SVM-RFE
* Random Forest

The final gene ranking is produced from the **mean of z-scored model outputs**.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

From the repository root:

```bash
streamlit run streamlit_app.py
```

The app expects a STRING interaction file with source-target columns such as:

* `#node1` and `node2`
* or other equivalent edge-column names handled by the pipeline

## Notebooks

Open the notebooks from the repository root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/ranking.ipynb
```

## Outputs

Typical generated outputs include:

* Topology feature tables
* Ranked gene lists
* Model evaluation summaries

Some generated CSV files already present in `data` are:

* [`data/string_topology_features.csv`](data/string_topology_features.csv)
* [`data/string_ml_gene_ranking.csv`](data/string_ml_gene_ranking.csv)
* [`data/string_topology_features_by_mcc.csv`](data/string_topology_features_by_mcc.csv)
* [`data/string_ml_gene_ranking_by_mcc.csv`](data/string_ml_gene_ranking_by_mcc.csv)

## Notes

* The Streamlit app is currently aligned with the median-label workflow from [`notebooks/ranking.ipynb`](notebooks/ranking.ipynb).
* MCC depends on clique structure, so it can become expensive on much larger networks.
* If you want to use a different STRING confidence threshold, filter the edge table before running the pipeline.
