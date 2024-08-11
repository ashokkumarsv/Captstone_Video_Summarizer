Text Summarization Using T5 Model
Overview

This repository contains the implementation of a text summarization model using the T5 (Text-To-Text Transfer Transformer) model. The project involves fine-tuning a pre-trained T5 model and comparing its performance with a custom-trained model. The model is evaluated using the ROUGE score, a widely used metric in text summarization tasks.
Features

    Fine-tuning a pre-trained T5 model on a custom dataset.
    Training a custom T5 model from scratch.
    Comparing the performance of the pre-trained and custom models using the ROUGE metric.
    Implementing User Acceptance Testing (UAT) for model validation.

Project Structure

kotlin

├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/
│   ├── pre_trained/
│   └── custom_trained/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── uat.ipynb
├── results/
│   ├── rouge_scores.csv
│   └── training_logs/
├── README.md
└── requirements.txt

    data/: Contains the training, validation, and test datasets.
    models/: Directory to save the fine-tuned pre-trained model and the custom-trained model.
    notebooks/: Jupyter notebooks for data preprocessing, model training, evaluation, and user acceptance testing.
    results/: Contains the results, including ROUGE scores and training logs.
    README.md: The file you're reading.
    requirements.txt: List of dependencies required to run the project.

Installation

    Clone the repository:

    bash

git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name

Create a virtual environment and activate it:

bash

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required dependencies:

bash

pip install -r requirements.txt

Install the necessary Python packages:

bash

    pip install transformers datasets rouge-score torch

Usage
Training the Models

    Pre-trained Model: Run the model_training.ipynb notebook to fine-tune the pre-trained T5 model on your dataset.
    Custom Model: Use the same notebook to train a custom T5 model from scratch.

Evaluating the Models

    Model Evaluation: Run the model_evaluation.ipynb notebook to compare the performance of the models using the ROUGE score.

User Acceptance Testing (UAT)

    UAT: The uat.ipynb notebook includes various testing scenarios and methodologies to ensure the model meets user requirements.

Results

The ROUGE scores for the models are stored in the results/rouge_scores.csv file. The pre-trained model achieved a ROUGE score of 0.763, while the custom model achieved a score of 0.4121.
Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss improvements.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

    Hugging Face Transformers
    ROUGE Score
