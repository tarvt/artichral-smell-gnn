# Detecting Microservice's Architectural Anti-Patterns Indicators Using Graph Neural Networks

## Table of Contents

1. [Project Overview](#project-overview)

2. [Folders Structure](#folders-structure)

   - [Dataset_Generation](#Dataset_Generation)

   - [Train](#Train)

   - [Simulator](#Simulator)

3. [Installation](#installation)

4. [Usage](#usage)

5. [Contact Information](#contact-information)

## Project Overview

This project focuses on the automatic detection of architectural anti-patterns in microservices using Graph Neural Networks (GNNs). Our approach aims to identify design issues early in the development process, providing developers with insights to improve the maintainability and scalability of their systems.

## Folders Structure

### Dataset

- **Description**: This folder contains the training and validation datasets along with the classes required to preprocess them for training.

- **Contents**:

  - Training and validation datasets

  - Preprocessing classes for training

### Simulator

- **Description**: This folder contains simulated code for MSNOSE and Arcan-Aroma.

- **Contents**:
  - Simulated code for MSNOSE and Arcan-Aroma

### Dataset_Generation

- **Description**: Contains the LLM prompts used to create graph datasets and the code that utilizes OpenAI to generate them.
- **Contents**:

  - LLM prompts for graph dataset creation
  - Code for using OpenAI to generate datasets

### Train

- **Description**: This folder includes subfolders for each GNN model we used; you can simply run any of them to train the model.
- **Contents**:
  - Subfolders for each GNN model
  - Training scripts per model

### Test_Loader

- **Description**: A folder that loads the final model, allowing you to test it using your own data.
- **Contents**:

  - Final model loading scripts
  - Testing scripts for custom datasets

## Installation

1.  Clone the repository:

    ```bash

    git clone https://github.com/tarvt/artichral-smell-gnn.git

    ```

2.  Set Up the Python Environment and Install Dependencies

3.  Training the Model

    ```bash

    cd train/graphsage
     python train_gnn.py

    ```

    The script will automatically:

        - Load the data using the MicroservicesDataset class defined in dataset.py.
        - Train the model using the parameters defined in the script.

# Usage

## Setup

First, ensure that you have Python 3.8 or later installed, along with PyTorch and PyTorch Geometric. Install the necessary packages using:

```bash
pip install torch torchvision torch-geometric

```

## Running the Model

To run the model on your data, follow these steps:

1. **Prepare Your Data:** Ensure your data is formatted correctly, according to the model's input requirements.

2. **Load and Run the Model:** Use the provided script to load your model and run it on the data.

## Test Model

### Overview

This section explains how to use the `test_model` directory to test the GraphSAGENet model with predefined or custom test data. The directory includes a Python script (`load_model_test.py`) and a sample data file (`test_data.json`) to help you quickly evaluate the model's performance on detecting architectural anti-patterns in microservices.

### Requirements

- Python 3.8 or later
- PyTorch 1.7 or later
- PyTorch Geometric
  Ensure all dependencies are installed using:

```bash

pip install torch torchvision torch-geometric
```

### Files in the Test Model Directory

- **load_model_test.py:** The main Python script for loading the model and running the tests.

- **test_data.json:** A JSON file containing sample test data formatted to match the expected input schema.

### Steps to Test the Model

1. **Navigate to the Test Model Directory:** Change your current working directory to `test_model` where the test script and data are located:

   ```bash

   cd test_model
   ```

### Review the Test Data

2. Open the `test_data.json` file to view the test data structure. You can modify this file to include your own test cases following the same format:

```json

{

  "Nodes": [[feature1, feature2, ...], ...],

  "edge_index": [[source1, source2, ...], [target1, target2, ...]],

  "edge_attr": [[attr1, attr2, ...], ...]

}
```

3. Execute the `load_model_test.py` script to load the model and test it on the data provided in `test_data.json`:

```bash

python load_model_test.py
```

### Interpret the Results

4. The script will output the detection results for each node.
