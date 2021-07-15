# Customer Propensity Modeling Project

##  Primary Objectives

We are trying to help a marketing executive understand which characteristics of a potential customer are best predictors of the propensity of potential customers to purchase one of the bank's products (in this example a term deposit). Besides a predictive model that would help target customers likely to buy the product, we want to be able to gain additional insight into the data through exploratory analysis, visualizations and feature assesments of important features and make suggestions to management based on this.

To summarize, we have been tasked with:
1. Marketing recommendations based on exploratory data analysis
2. Using a variety of data science techniques to implement predictive analytics on data
3. Technical Documentation for the methodology selected in the project that include diagrams of the ML pipeline and technical considerations made at each step of the pipeline. I will also include discussion of how we could deploy the selected model in a production environment

### Data Sources and General Description

We obtained data from the UCL repository (http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The dataset purports to record a direct marketing campaign of a Portuguese banking institution. It has 21 attributes, and 41188 instances with a binary label "y", delineated by a "yes" or "no", that describes if the individual had chosen to purchase the bank's product or not. 

Client information:

1 - age (numeric)

2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')

3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)

4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')

5 - default: has credit in default? (categorical: 'no','yes','unknown')

6 - housing: has housing loan? (categorical: 'no','yes','unknown')

7 - loan: has personal loan? (categorical: 'no','yes','unknown') related with the last contact of the current campaign:

8 - contact: contact communication type (categorical: 'cellular','telephone')

9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

other attributes:

12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

14 - previous: number of contacts performed before this campaign and for this client (numeric)

15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

social and economic context attributes

16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)

17 - cons.price.idx: consumer price index - monthly indicator (numeric)

18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)

19 - euribor3m: euribor 3 month rate - daily indicator (numeric)

20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):

21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

### Process flow

After importing the anticipated modules and libraries for this project, we will start with an arbitrary Exploratory Data Analysis that includes visualizations and written summaries, followed by a preprocessing and feature engineering process that takes into account findings from the analyses. We plan to fit a variety of machine learning models (Linear, Neighbors, and Ensemble) in multiple cases, as well as a Neural Network and then compare our results before a finalized recommendation.

![image](https://user-images.githubusercontent.com/47582640/125770033-e7d760dd-edda-4f78-81db-769462a439b2.png)


## Exploratory Data Analysis

### Initial Observations

There is a combination of categorical and numeric data. Based on description it looks like there is little to no missing data (null/unusable) so imputations are likely not required. We can do a further check for missing data. Results appear clean.

![image](https://user-images.githubusercontent.com/47582640/124596948-db103100-de30-11eb-8541-2875dba26189.png)

### Data Visualizations

We will start with visualizing numerical features. We will do histograms of all numeric data to get an idea of data spreads of continuous variable features. We follow this with a correlation Matrix to test for linear dependence within variables. shown are the first 4

![image](https://user-images.githubusercontent.com/47582640/124597008-ef542e00-de30-11eb-83dc-a7113dc97da6.png)

![image](https://user-images.githubusercontent.com/47582640/124597264-3fcb8b80-de31-11eb-9577-43d1722849b8.png)

Based on the correlation matrix we can see that there is heavy correlation between emp.var.rate and euribor3m. During feature engineering I plan to take out emp.var.rate as it also has a higher linear depencence with another 2 variables nr.employed and cons.price.idx

Next we can visualize categorical features, shown are the first 4:

![2](https://user-images.githubusercontent.com/47582640/124597742-d26c2a80-de31-11eb-98e7-cdbd6e69959e.png)
![3](https://user-images.githubusercontent.com/47582640/124597666-be282d80-de31-11eb-809b-5e94c6be97ca.png)

Based on looking at the bar plots which map out the results, there is a clear imbalance between the binary outcomes we are trying to classify. We need to visualize the ratio of yes and nos. There is significant, but understandable, discrepancy between the outcomes we are trying to classify. 88.7% of the outcomes from the marketing campaign suggest a no answer, an 8:1 ratio against yeses. We may need to take this account for the training process, possibly looking into SMOTE (Synthetic Minority Oversampling Technique) as part of the preprocessing pipeline.

![7](https://user-images.githubusercontent.com/47582640/124598009-270fa580-de32-11eb-8fd7-c5d180395764.png)

### EDA Summary

Summarizing our EDA, we get:
- There is an imbalance between the classes, with majority (88%) saying no. During preprocessing we can look into balancing classes using SMOTE to assist training and fitting process.
-  The Empvarrate (employment variation rate indicator) feature has a very high linear dependent relation with Euribor3m indicator feature (0.97). We can take this variable out of our training models.
- As expected calls under a certain duration (2-3 minutes) tend to almost certainly result in a no to purchasing. Something interesting is that there is also has a relation to the number of calls made to the customer. As number of calls increase we see the law of diminishing returns apply, as customers who get repeated calls tend to turn away from wanting to buy the product. A lot of time/money could be saved by not making repeated calls to customers who tend to end the phone calls before 3 minutes, and a limitation could be made to the number of calls made to the customer.

Some other observations we got on the demographics of who buy the product:
- There are statistically significant higher rates of success with contacting cellular phones than landlines.
- Divorced individuals tend to have a lower chance of buying the product
- Peope who have previously said yes tend to purchase the product

## Data Preprocessing

We need to do a couple of things for the preprocessing pipeline, proceeding from our data explorations:
1. Drop the Empvarrate column due to the heavy correlation with Euribor3m feature
2. Drop the Duration column feature to ensure that 
3. A resampling of the outcome frequencies to get a balanced classification in training (synthetic minority over-sampling, SMOTE)


## Running the Training Algorithms

We will use 4 different training models on each set of data, for a total of 8 initial training models, and compare the ROC/AUC curves:
- A Linear classifier, Logistic Regression
- A Neighbors classifier, KNeighbors
- A Tree Classifier, DecisionTree
- An Ensemble Classifier, Random Forest

We have enough samples available to run also run an artificial neural network, and see if we can get even better accuracies 

The best resulting model, Random Forest Classifier, is shown below. Results for others are summarized as well

![image](https://user-images.githubusercontent.com/47582640/125770437-558cc55d-b6ba-426b-b0e6-1bafe38ab616.png)


## Running an Artificial Neural Network

To compare the results of the classifier models used to a neural I designed a 3 layer Neural Network with the help of TensorFlow on processed data without SMOTE. I used 20 units on the first 2 layers to roughly match the features being utilized, with a sigmoid activation function within each neuron. The final layer gives a propensity between 0 and 1, and I will test out the results by classifying a propensity above 0.5 as being a yes, and under 0.5 as being a no. Because of the presence of under 50,000 samples, I anticipate this being a relatively quick and efficient training process for ad hoc analysis

The resulting neural network basically seems to roughly converge at roughly 0.90 accuracy around 20 epochs without being demanding on the GPU. We can test this out.

![image](https://user-images.githubusercontent.com/47582640/125771085-8ce42851-f9c0-43ef-8cab-117d1a615125.png)

## Feature Importance

We can use Recursive Feature Elimination (RFE) to see the impacts of most important features, ranked, on the best classifier model we tested out, an ensemble Random Forest Classifier using the numerical and categorical data.

![image](https://user-images.githubusercontent.com/47582640/125771322-e268c669-d0a5-4a86-a01c-51fc7dfd0df5.png)


# Summary of Results

Based on all our tests Model 8 (randomforest classifier model) has been the best in being able to predict the outcomes, having been trained with both the numeric and categorical data, and SMOTE applied. If we look at feature contributions to driving purchasing behavior, we can see that a lot of the most significant features are numeric.

The Deep learning neural network, used as benchmark, did not perform as well, likely due to the quantity of data. Recommendations will be made alongside EDA in the presentation attached.

