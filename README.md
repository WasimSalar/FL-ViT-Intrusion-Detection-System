# FL-ViT-Intrusion-Detection-System
Graduate Project

## Intrusion Detection System (IDS) for Autonomous Vehicles using Federated Learning and Vision Transformer

**Overview**

This repository contains the implementation of an Intrusion Detection System (IDS) designed to enhance the security of Federated Learning (FL)-based traffic sign recognition systems in autonomous vehicles. The IDS integrates FL with a Vision Transformer (ViT) architecture to detect and mitigate potential attacks on decentralized model training.

**Key Features:**

* **Federated Learning:** Enables collaborative model training across distributed devices while preserving data privacy.
* **Vision Transformer:** Leverages the power of Transformer architecture for robust traffic sign recognition.
* **Intrusion Detection:** Identifies and mitigates potential attacks on the FL-based system.
* **LISA Dataset:** Utilizes the LISA Traffic Sign Dataset for training and evaluation.

**Getting Started:**

1. **Set Up Environment:**
   - Install required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   -  Here's a breakdown of why each package is included:

        * **numpy:** Fundamental package for numerical computations.
        * **pandas:** Data analysis and manipulation tool.
        * **scikit-learn:** Machine learning library for various algorithms.
        * **matplotlib:** Plotting library for data visualization.
        * **tensorflow:** A popular deep learning framework (though not used extensively in the provided code, it might be useful for future extensions or alternative implementations).
        * **torch:** PyTorch, another popular deep learning framework, used extensively in the project.
        * **torchvision:** A collection of datasets, transforms, and models for computer vision, often used with PyTorch.
        * **logging:** For logging messages and debugging.
        * **Pillow:** Python Imaging Library, used for image processing.
        * **tqdm:** Progress bar library for monitoring training progress.
        * **matplotlib:** Plotting library for data visualization.
        * **os:** For operating system-related tasks like file and directory operations.
        * **json:** For working with JSON data.
   
   - Configure the LISA dataset path and other relevant settings in the configuration file.
2. **Train the Model:**
   - Run the training script:
     ```bash
     python train.py
     ```
3. **Evaluate the Model:**
   - Run the evaluation script:
     ```bash
     python main.py
     ```
4. **Deploy the IDS:**
   - Integrate the trained model into your autonomous vehicle system.
   - Configure the IDS to monitor network traffic and detect anomalies.


**Note on Notebook Structure:**

This notebook combines the essential components of the project, including data preprocessing, model architecture, training, and evaluation. By consolidating these elements into a single notebook, we can efficiently leverage the power of Kaggle's GPU resources.

**Key Components:**

* **Data Loading and Preprocessing:**
  - Loads the LISA Traffic Sign Dataset.
  - Preprocesses data using appropriate transformations.
  - Splits data into training and testing sets.
* **Model Architecture:**
  - Defines the Vision Transformer (ViT) architecture for image classification.
  - Configures the model's hyperparameters.
* **Federated Learning Implementation:**
  - Implements the federated learning framework, including:
    - Client-side training.
    - Server-side aggregation.
    - Communication rounds.
  - Incorporates the FedProx algorithm for improved convergence.
* **Training and Evaluation:**
  - Trains the model on the federated learning setup.
  - Evaluates the model's performance on the test set.
  - Visualizes training progress and results.

**Running the Notebook:**

1. **Upload the notebook to Kaggle.**
2. **Select a GPU runtime.**
3. **Run the cells sequentially.**

By following these steps, you can effectively execute the federated learning experiment and analyze the results.

**Note on Repository Structure:**

While the provided notebook combines all necessary components for a streamlined execution, it's recommended to maintain a well-organized project structure for better modularity and maintainability. 

**Suggested Repository Structure:**

```
FL-ViT-IDS/
├── clients/
│   └── clients.py
├── data/
│   └── data.py
├── LISA/
│   └── get_LISA.py
├── VisionTransformer
│   └── VisionTransformer.py
├── utils/
│   └── utils.py
├── scripts/
│   ├── train.py
│   └── main.py
├── requirements.txt
├── README.md
```

**Explanation:**

- **clients:** Defines functions related to managing the client-side operations in the federated learning process.
- **data:** Handles data loading, preprocessing, and potentially defines transformations for the training and testing datasets.
- **LISA:** Contains code for downloading and accessing the LISA dataset, a commonly used dataset for traffic sign recognition.
- **VisionTranformer:** Defines the Vision Transformer model architecture and its components used for image classification.
- **utils:** Contains utility functions for data loading, preprocessing, and evaluation.
- **scripts:** Contains scripts for training, evaluation, and main which serves as the entry point for the application.
- **requirements.txt:** Lists the required packages.
- **README.md:** Provides an overview of the project, instructions, and other relevant information.

This structure promotes code organization, reusability, and easier collaboration.

**Contributions:**
We welcome contributions to this project, including bug fixes, feature enhancements, and new ideas. Please feel free to open issues or pull requests.

**License:**
This project is licensed under the MIT License.
