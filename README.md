# deep-learning-challenge

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With the help of machine learning and neural networks, we use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

The Alphabet Soup’s business team provided a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

Dataset metadata:

- **EIN and NAME**: Identification columns
- **APPLICATION_TYPE**: Alphabet Soup application type
- **AFFILIATION**: Affiliated sector of industry
- **CLASSIFICATION**: Government organization classification
- **USE_CASE**: Use case for funding
- **ORGANIZATION**: Organization type
- **STATUS**: Active status
- **INCOME_AMT**: Income classification
- **SPECIAL_CONSIDERATIONS**: Special considerations for application
- **ASK_AMT**: Funding amount requested
- **IS_SUCCESSFUL**: Was the money used effectively

## Instructions

### Step 1: Preprocess the Data

Using Pandas and scikit-learn’s `StandardScaler()`, we preprocess the dataset in preparation for Step 2, where we compile, train, and evaluate the neural network model.

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, we design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset.

### Step 3: Optimize the Model

Continuing with TensorFlow, optimize the model to achieve a target predictive accuracy higher than 75%.

Options to optimize model:

- Dropping more or fewer columns.
- Creating more bins for rare occurrences in columns.
- Increasing or decreasing the number of values for each bin.
- Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs to the training regimen.

### Step 4: Write a Report on the Neural Network Model

Write an analysis report on the performance of the deep learning model.

## Overview of the Analysis

The goal of this project was to build a deep learning model for Alphabet Soup, a nonprofit organization, to predict whether applicants for funding would be successful in their ventures. Using a dataset of over 34,000 organizations, a binary classifier was created to predict the success of funded applicants, aiding Alphabet Soup in making more informed funding decisions.

## Results

### Data Preprocessing

1. What variable(s) are the target(s) for your model?
   
    **IS_SUCCESSFUL**: Was the money used effectively

1. What variable(s) are the features for your model?
   
   - **APPLICATION_TYPE**: Alphabet Soup application type
   - **AFFILIATION**: Affiliated sector of industry
   - **CLASSIFICATION**: Government organization classification
   - **USE_CASE**: Use case for funding
   - **ORGANIZATION**: Organization type
   - **STATUS**: Active status
   - **INCOME_AMT**: Income classification
   - **SPECIAL_CONSIDERATIONS**: Special considerations for application
   - **ASK_AMT**: Funding amount requested

2. What variable(s) should be removed from the input data because they are neither targets nor features?
    
    **EIN and NAME**: Identification columns

### Compiling, Training, and Evaluating the Model

1. How many neurons, layers, and activation functions did you select for your neural network model, and why?
   
   - Hidden Layer 1: 8 neurons, ReLU activation.
   - Hidden Layer 2: 24 neurons, ReLu activation.
   - Hidden Layer 3: 42 neurons, Sigmoid activation.
       
   This configuration was chosen after testing multiple setups using [TensorFlow Playground](https://playground.tensorflow.org/#activation=relu&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=8,8&seed=0.97853&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false) and observing that 2-3 hidden layers with at least 6-8 neurons provided strong results.
   
2. Were you able to achieve the target model performance?
   
   Initial model performance was 72.8%, which was below the target of 75%.

3. What steps did you take in your attempts to increase model performance?

    To increase model performance, I optimized the preprocessing step by keeping the NAME column (originally dropped along with EIN) and binning applicant names with fewer than 5 applications.  This resulted in 19803 applicants binned into an "Other" name category so that there would be less "rare" variables for the model to consider.

## Summary

The initial model achieved an accuracy of 72.8%, which was below the target of 75%.  After optimization steps, including keeping the NAME column and binning rare categories, the model’s performance improved to **79.1%** accuracy.

Optimization steps included:

- Retaining the NAME column and binning names with fewer than 5 occurrences into an “Other” category.
- Reducing the number of rare variables in the dataset, which helped simplify the model’s decision-making process.

While the optimized neural network model surpassed the 75% accuracy target, further improvements can be made using alternative models. I recommend exploring ensemble learning methods like Random Forests or Gradient Boosting Machines (GBM).  These methods are well-suited to many classification problems, including the one faced by Alphabet Soup.

Random Forests and GBMs are highly effective when dealing with datasets that contain many categorical features, like APPLICATION_TYPE, AFFILIATION, and CLASSIFICATION, which are present in the Alphabet Soup dataset.

If the target variable (IS_SUCCESSFUL) is imbalanced (i.e., if there are significantly more successful applicants than unsuccessful ones), ensemble methods tend to handle this imbalance better than standard neural networks. Random Forests use techniques like bootstrapping and random sampling, which help reduce bias and make the model more generalizable.

## Folders and Files

1. **AlphabetSoupCharity.ipynb:** Jupyter notebook with initial data preprocessing and model design.
2. **AlphabetSoupCharity_Optimization.ipynb:**  Updated Jupyter notebook with model optimization steps.
3. **AlphabetSoupCharity.h5**:  HDF file with results from initial notebook.
4. **AlphabetSoupCharity_Optimization.h5**:  HDF file with results from updated notebook.