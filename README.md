# Detecting Microservice's Architectural Anti-Patterns Indicators Using Graph Neural Networks

## Table of Contents

1. [Abstract](#abstract)

2. [Project Overview](#project-overview)

3. [Folders Structure](#folders-structure)

   - [Dataset](#dataset)

   - [Models](#models)

   - [Simulator](#simulator)

4. [Installation](#installation)

5. [Usage](#usage)

6. [Contributions](#contributions)

7. [License](#license)

8. [Acknowledgments](#acknowledgments)

9. [Contact Information](#contact-information)

## Abstract

Software systems must continuously evolve to meet new business needs. Today, the internet requires more flexible, scalable, and understandable software architecture. As a result, many companies and organizations have begun the process of migrating from monolithic architecture to more suitable architectures that can respond to current market demands. Microservices are small applications with a single responsibility that can be deployed, scaled, and tested independently. There are many advantages to using microservices, which is why they have recently become a popular topic.

However, their architecture is susceptible to inadequate solutions due to various factors such as time constraints, uncertainty, miscommunication, and increasing complexity of software systems. Such factors may lead to signs of bad architecture, and there is limited research on this issue. Identifying these architectural signs in microservices is important because they help in identifying potential design problems that can impact the overall quality and performance of the system. Early detection and resolution can make the system more maintainable and scalable.

To facilitate the identification of bad architectural signs in microservices, more empirical research needs to be conducted, and automated tools should be made available to developers to make this process more reliable and effective. However, there is a lack of public repositories that share patterns and practices of microservices in open-source projects. Identifying signs of bad microservice architecture is an issue that has received less attention. Therefore, the main goal of this project is to create a tool for the automatic detection of these signs in microservice architecture.

In this research effort, our primary objective is to develop a method for detecting bad signs in microservices using neural networks. In the context of detecting bad microservice signs, graph neural networks provide a powerful framework for analyzing interdependencies and interaction patterns in microservice architectures, effectively capturing complex relationships between entities. The results of this research indicate that, while similar tools provide comprehensive analyses using proven algorithms like depth-first search, they may not scale as well as newer techniques such as neural networks. Neural networks represent a promising alternative, especially for dynamic and large-scale microservice architectures, by optimizing computational resources.

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
