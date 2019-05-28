# Perceptron for Multiclass Classification

## Overview:

The goal of this project is to code, implement and evaluate Perceptron algorithm without using exiting libraries, and the code must be from scratch. Perceptron algorithm is selected to classify multiclass data of Owl types. The tools used for implementing algorithm is R studio, and the language used is R programming language.  Please refer to complete report in pdf format.

## Description:

Perceptron or single-layer neural network is the basic neural network. It receives multi-dimensional input which is processed using an activation function. The algorithm is trained using labeled data and learning algorithm then adjusts the weight in the process if there are any wrong predictions. It is a self-learning algorithm which uses back-propagation error method. In the self-learning phase, the difference between the predicted value and actual value is calculated. Based on this difference, the error is estimated. The error is back-propagated to all the units to keep the error at each unit proportional to the contribution of that unit towards total error of the process. This back-propagated error at each unit is used to optimize the weight at each connection.

<p align="center">
  <img width="600" height="400" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Perceptron.png" title=""> 
</p>                                         

## Design Decision:

Since one vs one approach is used, the data is divided into 3 groups with each group containing two species, and the type is labelled accordingly.

```r
                    shuffle_index <- createDataPartition(norm_data$Type, p = .66, list = FALSE)
                    TrainData <- norm_data[shuffle_index,]
                    TestData <- norm_data[-shuffle_index,]
    
                    Owl_type1_train <- TrainData[TrainData$Type == "LongEaredOwl",]
                    Owl_type2_train <- TrainData[TrainData$Type == "SnowyOwl",]
                    Owl_type3_train <- TrainData[TrainData$Type == "BarnOwl",]
    
                    ###########Creating Binary Groups
    
                    ####Pair_1##LongEared vs Snowvy Owl
    
                    Train_Grp_1 <- rbind(Owl_type1_train,Owl_type2_train) 
                    Train_Grp_1$Type <- ifelse((Train_Grp_1$Type == "LongEaredOwl"), 1, -1)
```
Euclidean norm or L2 norm is being calculated in the algorithm. It calculates the distance of the vector from the origin of the vector space. The L2 norm is further used to calculate the maximum norm of the vector which is used in the regularization of the neural network weights.

```r
                    euclidean_dist <- function(x){
                    sqrt(sum(x*x))
                    }    
    
                    Ecd_dist <- max(apply(x,1,euclidean_dist))
     
                    s <- euclidean_dist(w)
                    return(list(w= w/s, b = b/s, error = err, iteration = count))
```

The processing done at each neuron unit is denoted by:

<p align="center">
  <img width="400" height="100" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Code_Chunk_5.png"> 
</p>  

which is denoted in the code as:

```r
                    dist_plane <- function(z,w,b){
                    sum(z*w) + b
                    }
```

The weight(w) vector are the numerical parameters which shares the magnitude of impact each neuron has on another neuron. The input(z) vector is the vector of attributes on which the output depends which is the matrix multiplication to get the weighted sum.Bias(b) is analogous to the constant (c) which is added in the linear equation (y = m*x + c).  It is used to adjust the output of the weighted sum of the input.Activation function is the function to be applied on the output obtained from the neuron of the previous layer. The activation function applied in my algorithm is ‘heaviside step function’ which will label the output depending if the output is less than or greater than threshold value.

<p align="center">
  <img width="400" height="140" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Code_Chunk_7.png"> 
</p>  


```r
                    activte_fun <- function(x,w,b){
                    distances <- apply(x, 1, dist_plane,w,b)
                    return(ifelse(distances < 0, -1, +1))
                    }
```                             
The feature of perceptron is that it modifies the weight and bias initially provided as per the prediction obtained. If the prediction is wrong, then the weight and bias are updated with learning rate times. The learning rate helps in converging the model and deciding the appropriate weight and bias. The weight and bias keep on modifying until predicted ‘y’ is as close as possible to the actual ‘y’ value. The performance of the system depends solely on the distance between the actual ‘y’ and the predicted ‘y’.

<p align="center">
  <img width="400" height="300" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Code_Chunk_9.png"> 
</p>  
                                        
## Result:

The results are for one of the samples obtained by taking a random seed value.
         
<p align="center">
  <img width="400" height="300" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Result_Viz_1.png"> 
</p>
 

<p align="center"> 
  <img width="400" height="300" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Result_Viz_2.png">
</p>

<p align="center">
  <img width="800" height="600" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Result_Viz_3.png">
  </p>
  
<p align="center">
  <img width="800" height="450" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Result_Viz_4.png">
  </p>
  
<p align="center"> 
  <img width="400" height="150" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Result_Viz_5.png">
</p>
  
<p align="center"> 
  <img width="800" height="450" src="https://github.com/ak64119/R_perceptron/blob/master/Images/Result_Viz_6.png">
</p>
                                       
## Conclusion:
We can observe from ggplots that “LongEaredOwl” is linearly separable in both Body Length and Width and Wing Length and Width attributes as compared to "SnowyOwl" and "BarnOwl" which are close enough to mix at certain points. This complexity in data points in "SnowyOwl" and "BarnOwl" created problem in prediction of Owl Type in Test Data. As per the screen-shots above, the prediction of weight and bias in the data group of "SnowyOwl" and "BarnOwl" (p2) took maximum time, with the number of iterations reaching 11,043 and error count reaching 25,319 for the seed value 650 with 1 misclassification. The iterations/Epochs were not fixed to 1,000 to test the efficiency of the code in the test phase. The Epochs were later fixed to 1,000, and the accuracy for seed 650 moved down to 0.9555556 from 0.9777778 which is acceptable. If "SnowyOwl" and "BarnOwl” would be linearly separable, then the prediction would have been 100%. Therefore, the prediction of around 94% was achieved overall. We can conclude that the One vs One Perceptron approach works the best with linearly separable data, and it is computationally very expensive and very time consuming. The convergence is one of the biggest problems of the perceptron. It can be proved from the predictions that the perceptron learning rule converges if the two classes can be separated by linear hyperplane, but problems arise if the classes cannot be separated perfectly by a linear classifier as in the case of Train_Grp_2 of SnowyOwl vs BarnOwl…

## Reference:
1. Sagar,C. (2017). Creating & Visualizing Neural Network in R [online]. Available from: https://www.analyticsvidhya.com/blog/2017/09/creating-visualizing-neural-network-in-r/ [accessed 3 December 2018].
2. Sloth, R. (2014). Choosing a learning rate [online]. Available from: https://datascience.stackexchange.com/questions/410/choosing-a-learning-rate [accessed 26 November 2018].
3. Sharma, S. (2017). Epoch vs Batch Size vs Iterations [online]. Available from: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9 [accessed 18 November 2018].
4. Raschka, S. (2015). Single-Layer Neural Networks and Gradient Descent [online]. Available from: https://sebastianraschka.com/Articles/2015_singlelayer_neurons.html [accessed 20 November 2018].
5.  Lagandula, A. (2018). Perceptron Learning Algorithm: A Graphical Explanation Of Why It Works [online]. Available from: https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975 [accessed 23 November 2018].


 
 
 
 
 
