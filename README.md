# Intrusion Detection Model using Ensemble Learning

## Overview

This repository presents an optimized weighted ensemble learning approach for network intrusion detection, combining Convolutional Neural Networks (CNN), Autoencoders, and XGBoost. The model leverages CNN for spatial feature dependencies, Autoencoders for anomaly detection, and XGBoost for efficient attack classification. To handle class imbalance, techniques like SMOTE and ADASYN are used for better representation of minority attack classes. A weighted aggregation mechanism ensures optimal contribution from each model. The approach is evaluated on the CIC IDS 2017 dataset, achieving 99% accuracy and improved precision, recall, and F1-score, offering a scalable solution for cybersecurity applications.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn imbalanced-learn xgboost
```

## Dataset

- **Dataset Used**: CICIDS 2017
- **Classes**: BENIGN, DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS Slowloris, DoS Slowhttptest, Bot, Brute Force, Web Attack - XSS, Infiltration, Web Attack - SQL Injection, Heartbleed
- **Dataset Imbalance**: Severe
- **Preprocessing**: Basics done well
- Link for Dataset: https://www.unb.ca/cic/datasets/ids-2017.html

## Models Used

The ensemble learning approach integrates multiple models to enhance predictive performance. The selected models include:

1. **XGBoost**
2. **CNN**
3. **Autoencoder**

## Feature Engineering

- Applied feature engineering.
- Scaled numerical features using standard normalization techniques.
- Addressed class imbalance using resampling techniques if necessary.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author
Prithviraajan Senthilkumar

Licensed under [MIT](https://choosealicense.com/licenses/mit/)
