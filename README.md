# Detecting Microservice's Architectural Anti-Patterns Indicators Using Graph Neural Networks

## Table of Contents

1. [Project Overview](#project-overview)

2. [Folders Structure](#folders-structure)

   - [Dataset](#dataset)

   - [Models](#models)

   - [Simulator](#simulator)

3. [Installation](#installation)

4. [Usage](#usage)

5. [Contact Information](#contact-information)

## Project Overview

This project focuses on the automatic detection of architectural anti-patterns in microservices using Graph Neural Networks (GNNs). Our approach aims to identify design issues early in the development process, providing developers with insights to improve the maintainability and scalability of their systems.

## Folders Structure

### Dataset

- **Description**: This folder includes scripts and prompts used to generate microservice graphs using a large language model (LLM), specifically ChatGPT. The dataset serves as the training ground for our GNN models.

- **Contents**:

  - Graph generation scripts

  - Sample microservice graphs

  - Documentation on dataset generation

### Models

- **Description**: Contains the implementations of various Graph Neural Network models that learn from the generated dataset.

- **Contents**:

  - GNN architecture implementations

  - Training scripts

  - Model evaluation metrics and results

### Simulator

- **Description**: This folder contains simulated code for other tools to compare the performance and results of our GNN model against conventional techniques.

- **Contents**:

  - Simulation scripts

  - Benchmarking against existing tools

  - Documentation for usage

## Installation

1. Clone the repository:

   ```bash

   git clone https://github.com/tarvt/artichral-smell-gnn.git

   ```

# 5. Usage

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
