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
