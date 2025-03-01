# SVM Kernels Practical Implementation

## Overview
This project demonstrates the practical implementation of different Support Vector Machine (SVM) kernels for classification. The dataset is artificially created using circular patterns to explore the effectiveness of various kernel functions in handling non-linearly separable data.

## Libraries Used
- **`numpy`** – For numerical computations and data generation.
- **`pandas`** – For data handling and preprocessing.
- **`matplotlib.pyplot`** – For 2D data visualization.
- **`plotly.express`** – For interactive 3D visualizations.
- **`sklearn.model_selection.train_test_split`** – For splitting data into training and testing sets.
- **`sklearn.svm.SVC`** – For implementing Support Vector Classification.
- **`sklearn.metrics.accuracy_score`** – For evaluating model performance.

## Project Workflow
### 1. Data Generation
- Generates circular clusters of points for classification.
- Creates a DataFrame with features `X1` and `X2` and labels `Y`.
- Splits data into training and testing sets.

### 2. Implementing SVM with Linear Kernel
- Trains an SVM classifier with a **linear kernel**.
- Evaluates the classifier using **accuracy_score**.
- **Observation:** The linear kernel fails to correctly classify non-linearly separable data.

### 3. Polynomial Kernel Transformation
- Adds polynomial features (`X1^2`, `X2^2`, `X1*X2`) to transform the data.
- Visualizes the transformed data in a **3D space**.
- Trains a new **linear kernel SVM** on the transformed dataset.
- **Observation:** The polynomial transformation enables a linear decision boundary in 3D space.

### 4. Using RBF Kernel for Automatic Feature Mapping
- Trains an SVM classifier using **RBF kernel (`kernel='rbf'`)**.
- The RBF kernel automatically maps features into a higher-dimensional space.
- **Observation:** The RBF kernel achieves **perfect classification** (accuracy = 1.0).

## Results & Insights
- The **linear kernel** is ineffective for circular data separation.
- The **polynomial kernel** allows linear separation in an extended feature space.
- The **RBF kernel** provides automatic transformation and optimal classification.

## How to Run the Project
1. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn plotly
   ```
2. Run the Python script or Jupyter Notebook step by step.
3. Visualize the results and observe the decision boundaries.

## Future Improvements
- Implement **Sigmoid and Polynomial Kernels** for further comparisons.
- Tune **hyperparameters** for optimal model performance.
- Apply **SVM on real-world datasets** (e.g., image classification, sentiment analysis).

## Conclusion
This project highlights the significance of kernel functions in Support Vector Machines. The **RBF kernel** proves to be highly effective for non-linearly separable data, eliminating the need for manual feature engineering.
