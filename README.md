# Federated Learning for Multilingual Paraphrase Detection

This project simulates a federated learning system to train a single model for paraphrase detection in both **English** and **German** without centralizing the datasets. It extends the paraphrase detection research by Professor Pankaj Dadure by tackling the core challenge of training on **non-iid (statistically different)** data from multiple languages.

## 🛠️ Methodology

1.  **Sentence Embeddings:** A pre-trained `sentence-transformers` model (`all-MiniLM-L6-v2`) converts sentences into high-quality numerical vectors that capture semantic meaning.
2.  **Federated Averaging:** The simulation follows the standard Federated Averaging algorithm. In each round, a copy of the central global model is trained independently on each client's local data (English or German). The updated model weights are then sent back to the server and averaged to create an improved global model for the next round.

## 🚀 How to Run

The entire experiment is contained in a single Google Colab notebook.
1.  **Open the Notebook:** Launch the `.ipynb` file in Google Colab.
2.  **Enable GPU:** Go to `Runtime` -> `Change runtime type` and select `GPU`.
3.  **Run Cells:** Execute the notebook cells in order. Cell 2 is the most time-consuming as it generates the sentence embeddings.

## 📊 Results & Analysis

The experiment shows a clear performance increase in the initial rounds, proving the federated algorithm successfully aggregates knowledge from both non-iid clients. The model's performance converges after round 4, indicating it has reached a stable solution.

| Round | English Score | German Score |
| :---: | :-----------: | :----------: |
|   1   |    0.3551     |    0.3914    |
|   2   |    0.4063     |    0.4297    |
|   3   |    0.4217     |    0.4430    |
|   4   |    0.4235     |    0.4522    |
|   5   |    0.4167     |    0.4479    |
|   6   |    0.4168     |    0.4423    |
|   7   |    0.4094     |    0.4438    |
|   8   |    0.4123     |    0.4402    |
|   9   |    0.4068     |    0.4377    |
|  10   |    0.4141     |    0.4317    |

## 📈 Future Work & Continuous Improvement

The current results establish a strong baseline. The following steps will be taken for continuous performance improvement:

* **Advanced Algorithms:** Implement algorithms like **FedProx** or **SCAFFOLD** to better counteract client drift in non-iid settings.
* **Increase Training Duration:** Extend training to 50-100 rounds to allow for more fine-tuning.
* **Scale the Dataset:** Use the full PAWS and PAWS-X datasets.
* **Larger Models:** Experiment with more powerful multilingual models like `xlm-roberta-base`.
