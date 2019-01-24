# Xgboost
Xgboost implementation
XGBoost is an algorithm that has recently been dominating applied machine learning and Kaggle competitions for structured or tabular data.

XGBoost is an implementation of gradient boosted decision trees designed for speed and performance.

In this post you will discover XGBoost and get a gentle introduction to what is, where it came from and how you can learn more.

After reading this post you will know:

What XGBoost is and the goals of the project.
Why XGBoost must be apart of your machine learning toolkit.
Where you can learn more to start using XGBoost on your next machine learning project.
Let’s get started.



The name xgboost, though, actually refers to the engineering goal to push the limit of computations resources for boosted tree algorithms. Which is the reason why many people use xgboost.

— Tianqi Chen, in answer to the question “What is the difference between the R gbm (gradient boosting machine) and xgboost (extreme gradient boosting)?” on Quora

It is an implementation of gradient boosting machines created by Tianqi Chen, now with contributions from many developers. It belongs to a broader collection of tools under the umbrella of the Distributed Machine Learning Community or DMLC who are also the creators of the popular mxnet deep learning library.

Tianqi Chen provides a brief and interesting back story on the creation of XGBoost in the post Story and Lessons Behind the Evolution of XGBoost.

XGBoost is a software library that you can download and install on your machine, then access from a variety of interfaces. Specifically, XGBoost supports the following main interfaces:

Command Line Interface (CLI).
C++ (the language in which the library is written).
Python interface as well as a model in scikit-learn.
R interface as well as a model in the caret package.
Julia.
Java and JVM languages like Scala and platforms like Hadoop.
XGBoost Features
The library is laser focused on computational speed and model performance, as such there are few frills. Nevertheless, it does offer a number of advanced features.

Model Features
The implementation of the model supports the features of the scikit-learn and R implementations, with new additions like regularization. Three main forms of gradient boosting are supported:

Gradient Boosting algorithm also called gradient boosting machine including the learning rate.
Stochastic Gradient Boosting with sub-sampling at the row, column and column per split levels.
Regularized Gradient Boosting with both L1 and L2 regularization.
System Features
The library provides a system for use in a range of computing environments, not least:

Parallelization of tree construction using all of your CPU cores during training.
Distributed Computing for training very large models using a cluster of machines.
Out-of-Core Computing for very large datasets that don’t fit into memory.
Cache Optimization of data structures and algorithm to make best use of hardware.
Algorithm Features
The implementation of the algorithm was engineered for efficiency of compute time and memory resources. A design goal was to make the best use of available resources to train the model. Some key algorithm implementation features include:

Sparse Aware implementation with automatic handling of missing data values.
Block Structure to support the parallelization of tree construction.
Continued Training so that you can further boost an already fitted model on new data.
XGBoost is free open source software available for use under the permissive Apache-2 license.

Why Use XGBoost?
The two reasons to use XGBoost are also the two goals of the project:

Execution Speed.
Model Performance.
1. XGBoost Execution Speed
Generally, XGBoost is fast. Really fast when compared to other implementations of gradient boosting.

Szilard Pafka performed some objective benchmarks comparing the performance of XGBoost to other implementations of gradient boosting and bagged decision trees. He wrote up his results in May 2015 in the blog post titled “Benchmarking Random Forest Implementations“.

He also provides all the code on GitHub and a more extensive report of results with hard numbers.

Benchmark Performance of XGBoost
Benchmark Performance of XGBoost, taken from Benchmarking Random Forest Implementations.

His results showed that XGBoost was almost always faster than the other benchmarked implementations from R, Python Spark and H2O.

From his experiment, he commented:

I also tried xgboost, a popular library for boosting which is capable to build random forests as well. It is fast, memory efficient and of high accuracy

— Szilard Pafka, Benchmarking Random Forest Implementations.

2. XGBoost Model Performance
XGBoost dominates structured or tabular datasets on classification and regression predictive modeling problems.

The evidence is that it is the go-to algorithm for competition winners on the Kaggle competitive data science platform.

For example, there is an incomplete list of first, second and third place competition winners that used titled: XGBoost: Machine Learning Challenge Winning Solutions.

To make this point more tangible, below are some insightful quotes from Kaggle competition winners:

As the winner of an increasing amount of Kaggle competitions, XGBoost showed us again to be a great all-round algorithm worth having in your toolbox.

— Dato Winners’ Interview: 1st place, Mad Professors

When in doubt, use xgboost.

— Avito Winner’s Interview: 1st place, Owen Zhang

I love single models that do well, and my best single model was an XGBoost that could get the 10th place by itself.

— Caterpillar Winners’ Interview: 1st place

I only used XGBoost.

— Liberty Mutual Property Inspection, Winner’s Interview: 1st place, Qingchen Wang

The only supervised learning method I used was gradient boosting, as implemented in the excellent xgboost package.

— Recruit Coupon Purchase Winner’s Interview: 2nd place, Halla Yang

What Algorithm Does XGBoost Use?
The XGBoost library implements the gradient boosting decision tree algorithm.

This algorithm goes by lots of different names such as gradient boosting, multiple additive regression trees, stochastic gradient boosting or gradient boosting machines.

Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. A popular example is the AdaBoost algorithm that weights data points that are hard to predict.

Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.

This approach supports both regression and classification predictive modeling problems.
