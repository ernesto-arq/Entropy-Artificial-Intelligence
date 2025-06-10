# Experiments-with-entropy-and-Artificial-Intelligence
Experiments with entropy and Artificial Intelligence


**Overview**  
This repository contains Python scripts for statistical analysis that focus on Gaussian distributions, entropy measurements in geometric contexts, and analysis of centralized data with independent and identically distributed (iid) and non-independent and non-identically distributed (non-iid) characteristics. These tools were developed for data scientists and researchers looking to investigate statistical models and data behavior in a variety of scientific and technological contexts.

**Contents**  
1. Gaussian analysis    
File: gaussian_analysis.ipynb  
Description: This script performs detailed analysis using Gaussian distributions. It includes functionality for fitting models to data, estimating parameters, and evaluating the statistical properties of Gaussian-based models.  

2. Geographic analysis of entropy  
File: geometric_analysis_of_entropy.ipynb  
Description: Dedicated to the study of entropy within geometric data structures. This script calculates entropy measurements and explores their implications in data complexity and information theory.

2.1 Here is a brief and straightforward description of each script within the directory (IDD datagen and non-IDD datagen):  
* cifar100_noniid.py: This script generates non-IID data partitions for the CIFAR-100 dataset, supporting distributed machine learning experiments with varied data distribution across clients.  
* cifar10_noniid.py: Tailored for the CIFAR-10 dataset, this script creates non-IID subsets for diverse model training needs.  
* fashionmnist_noniid.py: Manages the partitioning of the Fashion-MNIST dataset into non-IID subsets, enhancing model testing under various data distributions.  
* mnist_noniid.py: Facilitates the creation of non-IID data partitions for the MNIST dataset, aiding in algorithm testing across different data characteristics.  
* create_MNIST_datasets.py, create_MNIST_datasets1.py, create_MNIST_datasets2.py: These scripts generate customized MNIST datasets with potential variations in non-IID configurations or noise injections to mirror real-world data scenarios.

3. IID Centralized Data  
File: iid_centralized_data.ipynb   
Description: Analyzes centralized data under the assumption that it is iid. This script is useful for statistical inference, providing insights into data patterns and distribution without the influence of data correlations.

5. non-IID Federated Learning Distributed Data  
File: non_IID_Fedlearning.ipynb  
This project demonstrates the implementation of a federated learning system using PyTorch to train a Convolutional Neural Network (CNN) in a distributed environment. Training is carried out locally on several clients that do not share data among themselves non-IID, using federated averaging to combine the local models into a robust global model.

**Getting Started**  
To use these scripts:  

Clone the repository using git clone [repository-url].  
Ensure that you have Python installed on your machine. These scripts were tested on Python 3.8.  
Install necessary dependencies by running pip install -r requirements.txt (Note: you must create this file based on the libraries used in the scripts).  

**Usage**
To run any of the scripts, navigate to the repository directory and execute:  
python <script_name.py>  
Replace <script_name.py> with the name of the script you wish to run.  

**Contributing**  
Contributions to this repository are welcome. Please follow these steps to contribute:  
Fork the repository.  
1. Create your feature branch (git checkout -b feature/YourFeatureName).  
2. Commit your changes (git commit -am 'Add some feature').  
3. Push to the branch (git push origin feature/YourFeatureName).  
4. Open a new Pull Request.  


**License**  
This project is licensed under the MIT License - see the LICENSE file for details.  
[MIT License](https://opensource.org/licenses/MIT)

**Contact**  
For any queries regarding this repository, please open an issue in the repository or contact the repository administrators directly.  
This README template provides a formal and comprehensive introduction to your repository. If you need further customization or additional sections, feel free to let me know! 
E-mail: gurgelvalente@alu.ufc.br

**General Information**  

Note: This analysis does not focus on information security, bandwidth management, latency, or client availability. Therefore, the simulation aims to control and ensure reproducibility in a federated learning scenario where devices and client availability are constantly changing. It seeks to isolate the variables of network security, client availability, and latency, focusing on data distribution and analysis.

1. Fundamental Structure of Federated Learning: The simulation architecture implements the fundamental functionalities of a federated environment, focusing on enabling detailed, data-centric analysis in the laboratory. The functionalities include, along with the design, characteristics of imbalanced data, seeking to approximate more realistic scenarios. The implementation incorporates more complex functionalities present in works by other authors, including:
* Furthermore, the design was chosen for cost and scalability reasons, as the complexity and costs associated with acquiring, deploying, and managing multiple edge devices made research on data properties and characteristics more accessible. Iteration speed also increases, allowing multiple experiments to be performed on the data and architecture with different hyperparameters.
* In this way, the simulation architecture aims to represent the main characteristics of a federated environment related to data and client heterogeneity, including imbalanced data, allowing for the observation of results when stressing the algorithms and analyses, as well as performing various observations.

2. Key Components and Functionalities
* Data Distribution among Clients: The simulation of data heterogeneity is performed using functions built based on the literature and the authors' codes, such as MNISTNonIID, FashionMNISTNonIID, Cifar10NonIID, and Cifar100NonIID, giving due credit for the ideas. The functions were also adapted for other datasets to allow a more realistic data distribution.
* Local Training: The train_client function performs model training, virtually simulating each client in tensors. The goal is to represent, in a decentralized and virtual way, federated learning, where the raw data of each client remains locally.
* Model Aggregation: The aggregate_models function implements the representative method of federated learning, in which each client updates the model weights and sends these weights to the aggregation stage.
* Global Model Evaluation: After aggregation is complete, the test_model evaluation function is called to measure the performance of the global model. This allows for evaluation at each new aggregation to understand the nature of the data and its effects.
* Advanced Features and Virtualization:Non-IID Data Simulation: The main feature of the code is its ability to simulate non-IID environments. It has specific functions to create unbalanced distributions (MNISTNonIIDunbalance) and with different classes per client.
* Non-IID Data Function Import: Functions from the literature are used to create unbalanced distributions, approximating the behavior of the data to reality.
* Aggregation Modularity: The framework allows evaluating different aggregation algorithms, such as FedAvg and FedProx (for example, via train_client_fedprox). The design is adaptable, allowing changes and the addition of new algorithms.
* Orchestration and Control: The primary function represents a central orchestrator of federated learning, managing the training process and distributing the global model to clients. It also decides which methods clients should use and collects the results for visualization and critical analysis of the process.

3. Advanced Features
* Evaluation Metrics: Performance metrics are integrated for both the global model and the clients through functions such as plot_metrics, plot_client_metrics, and plot_confusion_matrix, allowing for a detailed analysis of data behavior.
* It is essential to note that this simulation is a model of reality, acknowledging the limitations inherent in real-world experiences. In addition, the simulation does not cover:
* Hardware heterogeneity: The simulation does not consider different hardware.
* Network communication: Factors such as network latency, communication failures, or clients who decide to abandon the training are not taken into account in the experiment.
* Client behavior: The simulation assumes that clients will complete the assigned task when ordered to train. Events such as battery failure, power failure, network outage, loss of connection, decision to abandon the training process, or damage to the equipment that prevents communication and task execution during the activity are not considered.

**Important note**  
Changing the logarithmic base may increase performance on some models (granularity of description).
