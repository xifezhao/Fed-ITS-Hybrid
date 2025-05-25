# Fed-ITS-Hybrid: Personalized Federated Learning with Hierarchical Models and Dynamic Client Clustering

This repository implements "Fed-ITS-Hybrid," a federated learning (FL) framework designed to handle data heterogeneity across clients by employing a hierarchical model structure and dynamic client clustering. The framework decomposes client models into global, cluster-specific, and personalized components, allowing for more nuanced adaptation to local data distributions.

## Overview

Traditional Federated Learning (e.g., FedAvg) often struggles when client data is non-IID (Independent and Identically Distributed). Fed-ITS-Hybrid addresses this by:

1.  **Hierarchical Model Structure:** Each client's model is a combination of:
    *   **Global Model (GB):** Learns common features shared across all clients.
    *   **Cluster-Specific Model (CS):** Learns features shared among a sub-group (cluster) of clients.
    *   **Personalized Head (PH):** Learns features unique to an individual client.
2.  **Dynamic Client Clustering:** Clients are periodically clustered based on the similarity of their personalized model components (PH). This allows the system to adaptively group clients with similar data characteristics or learning tasks.
3.  **Synthetic Data Generation:** The script includes a flexible synthetic data generator that can create data with controllable global, cluster-specific, and personal effects, supporting both linear and non-linear relationships.
4.  **Baseline Comparisons:** The performance of Fed-ITS-Hybrid is compared against:
    *   **Standard FedAvg:** A single global model for all clients.
    *   **Local Training Only (Structured PH):** Clients train only their personalized heads, assuming fixed (zeroed) global and cluster components, to simulate a purely personalized scenario within the hierarchical structure.

## Key Features

*   **Hierarchical Model Architecture:** GB + CS + PH components for flexible personalization.
*   **Dynamic Clustering:** Uses K-Means on flattened personalized model parameters to group clients.
*   **Configurable Synthetic Data:** Generate data with varying levels of global, cluster, and personal influences, with options for non-linear transformations.
*   **Linear or MLP Models:** Supports both simple linear models and Multi-Layer Perceptrons (MLPs) for all model components.
*   **Comprehensive Evaluation:** Tracks Mean Squared Error (MSE) for performance and Adjusted Rand Index (ARI) for clustering quality (if true cluster labels are known).
*   **Detailed Visualizations:**
    *   Performance comparison curves (MSE vs. Communication Rounds).
    *   ARI scores over rounds for Fed-ITS-Hybrid.
    *   t-SNE visualization of personalized head (PH) parameters, colored by learned and true clusters.
    *   Per-cluster average loss evolution for Fed-ITS-Hybrid.
    *   (Optional) Contribution of GB, CS, and PH components to a specific client's prediction.
    *   (Optional) Visualization of the 1D personalized head (PH) function if `NUM_FEATURES_PERSONAL` is 1.

## Dependencies

The script requires the following Python libraries:

*   Python 3.7+
*   PyTorch (`torch`)
*   NumPy (`numpy`)
*   Scikit-learn (`sklearn`)
*   SciPy (`scipy`)
*   Matplotlib (`matplotlib`)

You can install them using pip:

```bash
pip install torch numpy scikit-learn scipy matplotlib
