# **Knowledge Graph Driven Drug Recommendation for Breast and Cervical Cancer**

This repository contains the implementation of a diffusion-based framework for refining medical knowledge graphs (KGs) and phenotype-drug interaction graphs, enabling accurate and context-aware drug recommendations.

---

## **Repository Structure**

### **1. Codes**
The `codes` folder contains the main implementation of the framework:
- **Main Code**: Includes the implementation of:
  - **Diffusion Process**: To refine and denoise the knowledge graph.
  - **Collaborative Filtering**: To optimize phenotype-drug interactions using Bayesian Personalized Ranking (BPR) Loss.
  - **Graph Neural Networks (GNN)**: Implements GCN, GAT, and GIN architectures for embedding learning.
  - **Joint Optimization**: Combines BPR Loss, Knowledge Graph Loss, and Generative Model Loss.

### **2. Datasets**
The `datasets` folder includes the input data files used for training and evaluation:
- **PharmKG**: A domain-specific knowledge graph containing drug-disease and disease-pathway relationships.
- **PrimeKG**: A comprehensive medical knowledge graph used for relational enrichment and embedding refinement.

### **3. Results**
The `results` folder contains:
- **Performance Metrics**: Results of the evaluated GNN architectures (GCN, GAT, and GIN) on Recall@20 and NDCG@20.
- **Figures and Plots**: Visualization of performance metrics, training progress, and loss convergence.
- **Ablation Study**: Results demonstrating the impact of Diffusion Module (DM) and Cross-Knowledge Graph Consistency (CKGC) on model performance.

---

## **How to Use**


Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/agnik2019/Knowledge_Graph_Driven_Drug_Recommendation
cd codes
2. Running the Code
```
To train the model and evaluate results, run:

python main.py
The configuration options for learning rate, batch size, number of epochs, and GNN type can be set in the argument parser.

- Data Preparation
Place the PharmKG.csv and PrimeKG.csv files in the datasets folder.
The code will automatically load and process these files during execution.
## Key Results
Best Model: GCN with Diffusion Module (DM) achieved:

- Recall@20: 55.51%
- NDCG@20: 25.26%
### Ablation Study:

Demonstrated the importance of the Diffusion Module and Cross-Knowledge Graph Consistency (CKGC) in improving performance.

## Contributing
Feel free to submit issues or create pull requests for improvements and additional features.



## Acknowledgments
This work utilizes datasets from PharmKG and PrimeKG and the methodology from DiffKG ([diffkg_link](https://github.com/HKUDS/DiffKG)).
