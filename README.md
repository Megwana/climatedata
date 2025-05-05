# Climate Change Temperature Anomaly Analysis

## Overview

This program performs an analysis of global temperature anomaly data, focusing on both linear and polynomial regression models to predict future temperature trends. It uses data visualisations to help interpret the results and is designed to be user-friendly, providing clear feedback and interaction.

## Features

### **Data Loading & Cleaning**
- **Loading the Data**:
  - The program allows for dynamic loading of data from a user-provided file path.
  - Filters out data specific to "Temperature" anomalies using string matching in the "Indicator" column.
  
- **Data Transformation**:
  - Converts data from a wide to a long format for better handling (using `melt()`).
  - Handles missing values through mean imputation, ensuring that no missing values interfere with analysis.
  - Detects and removes outliers based on z-scores, ensuring data integrity and accuracy.

### **Exploratory Data Analysis (EDA)**
- **Data Exploration**:
  - The program visually presents trends in global temperature anomalies over time, making it easier to identify patterns.
  - It prints the global temperature anomaly trend derived from the linear regression model, providing a straightforward interpretation.

### **Model Building**
- **Linear Regression Model**:
  - Trains a **linear regression model** on the cleaned dataset to predict global temperature anomaly trends over time.
  - Justifies the choice of linear regression for its ability to model simple linear relationships and trends in the data.
  - Outputs the model's coefficient, allowing users to interpret the long-term trend.

- **Polynomial Regression Model**:
  - Trains a **polynomial regression model** to capture non-linear relationships in the data that might be missed by a simple linear model.
  - Allows the user to choose the degree of the polynomial, adding flexibility for more complex trend modelling.
  - Justifies the use of polynomial regression for more accurate modelling of non-linear data trends.

### **Model Evaluation & Predictions**
- **Prediction Visualisation**:
  - The code visualises future predictions of global temperature anomalies for the years 2023-2035 based on both linear and polynomial regression models.
  - **Line plots** are generated for both models to compare the predicted temperature anomaly trends visually, making the results interpretable to users.
  - These visualisations are saved as images, ensuring that the output can be easily shared or referenced later.

- **Future Predictions**:
  - The future predictions are based on the trained models, allowing for the forecasting of future global temperature trends.

### **Interactivity & User Experience**
- **Interactive Command-Line Interface (CLI)**:
  - Provides an easy-to-use menu-driven interface where users can:
    - Load data,
    - Train the linear and polynomial regression models,
    - Visualise the predictions,
    - Exit the program.
  - Guides users through each step of the analysis, improving usability.

- **Dynamic File Loading**:
  - The program allows users to specify their data file, making it flexible and adaptable to different datasets.

### **Justification of Model Choices**
- **Clear Justification for Model Selection**:
  - The program explains the rationale behind choosing **linear regression** (for modelling long-term linear trends) and **polynomial regression** (to capture non-linear patterns).
  - Provides users with a better understanding of model assumptions and how they apply to climate data.

### **Code Efficiency & Organisation**
- **Modular Code Structure**:
  - Code is well-organised into distinct functions for different tasks (data loading, cleaning, model training, and plotting).
  - The modular structure ensures that the program is maintainable, understandable, and extendable.

- **Comments & Documentation**:
  - Functions are properly documented with clear docstrings explaining their purpose and parameters.
  - This enhances the readability of the code and ensures that others can easily understand and modify the program.

### **Visual Outputs**
- **Data Visualisations**:
  - Line plots are generated to show the future predicted anomalies for both the linear and polynomial models.
  - These visual outputs help make the findings accessible and interpretable for users, especially those without deep technical expertise.

### **File Management**
- **Saving Visual Outputs**:
  - Saves the generated plots as PNG files, making it easy for users to store or present the visualisations in reports or presentations.

### **User Guidance & Feedback**
- **User-Friendly Prompts**:
  - Provides users with informative prompts at each step (e.g., asking for file paths, providing feedback on successful data loading, etc.).
  - Offers a clear explanation of the models used and what the user can expect from the results.

### **Error Handling & Robustness (Future Considerations)**
- While not fully implemented in the current version, the use of `try-except` blocks can be added to improve the robustness of the code, ensuring it gracefully handles errors (e.g., invalid file paths, missing data).

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
