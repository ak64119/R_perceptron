# Perceptron for Multiclass Classification

## Overview:

The goal of this project is to code, implement and evaluate Perceptron algorithm without using exiting libraries, and the code must be from scratch. Perceptron algorithm is selected to classify multiclass data of Owl types. The tools used for implementing algorithm is R studio, and the language used is R programming language.  Please refer to complete report in pdf format.

## Description:

Perceptron or single-layer neural network is the basic neural network. It receives multi-dimensional input which is processed using an activation function. The algorithm is trained using labeled data and learning algorithm then adjusts the weight in the process if there are any wrong predictions. It is a self-learning algorithm which uses back-propagation error method. In the self-learning phase, the difference between the predicted value and actual value is calculated. Based on this difference, the error is estimated. The error is back-propagated to all the units to keep the error at each unit proportional to the contribution of that unit towards total error of the process. This back-propagated error at each unit is used to optimize the weight at each connection.
                                       
## Design Decision:

Since one vs one approach is used, the data is divided into 3 groups with each group containing two species, and the type is labelled accordingly.
                             
Euclidean norm or L2 norm is being calculated in the algorithm. It calculates the distance of the vector from the origin of the vector space. The L2 norm is further used to calculate the maximum norm of the vector which is used in the regularization of the neural network weights.
                                    
                      
               
The processing done at each neuron unit is denoted by:
                                      
which is denoted in the code as:
                                      
The weight(w) vector are the numerical parameters which shares the magnitude of impact each neuron has on another neuron. The input(z) vector is the vector of attributes on which the output depends which is the matrix multiplication to get the weighted sum.Bias(b) is analogous to the constant (c) which is added in the linear equation (y = m*x + c).  It is used to adjust the output of the weighted sum of the input.Activation function is the function to be applied on the output obtained from the neuron of the previous layer. The activation function applied in my algorithm is ‘heaviside step function’ which will label the output depending if the output is less than or greater than threshold value.
                                              

                               
The feature of perceptron is that it modifies the weight and bias initially provided as per the prediction obtained. If the prediction is wrong, then the weight and bias are updated with learning rate times. The learning rate helps in converging the model and deciding the appropriate weight and bias. The weight and bias keep on modifying until predicted ‘y’ is as close as possible to the actual ‘y’ value. The performance of the system depends solely on the distance between the actual ‘y’ and the predicted ‘y’.
                                        
## Result:

The results are for one of the samples obtained by taking a random seed value.
         
   
 
 
 
 
 
                                         
                                                
## Conclusion:
We can observe from ggplots that “LongEaredOwl” is linearly separable in both Body Length and Width and Wing Length and Width attributes as compared to "SnowyOwl" and "BarnOwl" which are close enough to mix at certain points. This complexity in data points in "SnowyOwl" and "BarnOwl" created problem in prediction of Owl Type in Test Data. As per the screen-shots above, the prediction of weight and bias in the data group of "SnowyOwl" and "BarnOwl" (p2) took maximum time, with the number of iterations reaching 11,043 and error count reaching 25,319 for the seed value 650 with 1 misclassification. The iterations/Epochs were not fixed to 1,000 to test the efficiency of the code in the test phase. The Epochs were later fixed to 1,000, and the accuracy for seed 650 moved down to 0.9555556 from 0.9777778 which is acceptable. If "SnowyOwl" and "BarnOwl” would be linearly separable, then the prediction would have been 100%. Therefore, the prediction of around 94% was achieved overall. We can conclude that the One vs One Perceptron approach works the best with linearly separable data, and it is computationally very expensive and very time consuming. The convergence is one of the biggest problems of the perceptron. It can be proved from the predictions that the perceptron learning rule converges if the two classes can be separated by linear hyperplane, but problems arise if the classes cannot be separated perfectly by a linear classifier as in the case of Train_Grp_2 of SnowyOwl vs BarnOwl…

## Reference:
https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/
https://datascience.stackexchange.com/questions/410/choosing-a-learning-rate
https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html
https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975
Book: Neural Network with R by Giuseppe and Balaji

 
 
 
 
 
