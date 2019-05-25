#########Downloading important packages
#install.packages("ggplot2")
library("ggplot2")
#install.packages("caret")
library("caret")
#########Upload Raw Dataset and name the columns

#getwd()

setwd("E:/New Volume/Academic/NUIG_College/ML/Assignment_3/")

Data_Set <-  read.csv("owls.csv", header = FALSE)

colnames(Data_Set) <- c("Body_Length","Wing_Length", "Body_Width", "Wing_Width", "Type")


Classification_Algo <- function(Data_Set, no_of_iterations){
  
  raw_data <- Data_Set
  cycles <- no_of_iterations
  
  ##########Normalizing Data
  
  normzng_functn <- function(x){
    ((x - min(x))/(max(x) - min(x)))
  }
  
  raw_data$Type <- as.factor(raw_data$Type)
  norm_data <- as.data.frame(apply(raw_data[,-5], 2, normzng_functn))
  norm_data$Type <- raw_data$Type
  
  #summary(norm_data)
  
  ##########Initial Data Exploration
  
  ggplot(raw_data, aes(x = Body_Length, y = Body_Width)) + 
    geom_point(aes(colour=Type, shape=Type), size = 3) +
    xlab("Body Length") + 
    ylab("Body Width") + 
    ggtitle("Body Length V/s Body Width as per Owl Type")
  
  ggplot(raw_data, aes(x = Wing_Length, y = Wing_Width)) + 
    geom_point(aes(colour=Type, shape=Type), size = 3) +
    xlab("Wing Length") + 
    ylab("Wing Width") + 
    ggtitle("Wing Length V/s Wing Width as per Owl Type")
  
  
  ###########Preparing Training and Test Data
  
  Accuracy_list <- vector(mode = 'numeric', length = cycles)
  
  for (k in 1:cycles){
    set.seed(340 + (70*k))   #Random calculation to calculate random value of k
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
    
    Train_Grp_1_data <- Train_Grp_1[,(1:4)]
    Train_Grp_1_exptd <-Train_Grp_1[, 5]
    
    ####Pair_2##SnowyOwl vs BarnOwl
    
    Train_Grp_2 <- rbind(Owl_type2_train,Owl_type3_train) 
    Train_Grp_2$Type <- ifelse((Train_Grp_2$Type == "SnowyOwl"), 1, -1)
    
    Train_Grp_2_data <- Train_Grp_2[,(1:4)]
    Train_Grp_2_exptd <-Train_Grp_2[, 5]
    
    ####Pair_3##Barn_Owl Vs LongEaredOwl
    
    Train_Grp_3 <- rbind(Owl_type3_train,Owl_type1_train) 
    Train_Grp_3$Type <- ifelse((Train_Grp_3$Type == "BarnOwl"), 1, -1)
    
    Train_Grp_3_data <- Train_Grp_3[,(1:4)]
    Train_Grp_3_exptd <-Train_Grp_3[, 5]
    
    ###########Creating Perception Train and Test Algorithm
    
    #Calculate euclidean distance
    euclidean_dist <- function(x){
      sqrt(sum(x*x))
    }
    
    #Calculate distance from plane
    dist_plane <- function(z,w,b){
      sum(z*w) + b
    }
    
    #Function to assign value based on value
    activte_fun <- function(x,w,b){
      distances <- apply(x, 1, dist_plane,w,b)
      return(ifelse(distances < 0, -1, +1))
    }
    
    #Perceptron function to train the model
    perceptron <- function(x,y,learn_rate = 1){
      w <- rep(0, length = ncol(x))  #Initial weight
      b <- 0 #Initialize bias
      count <- 0 #track the run count
      err <- rep(0,1000) #count update of error
      Ecd_dist <- max(apply(x,1,euclidean_dist))
      flag <- TRUE
      
      #Back-propagation to adjust the weights to
      #keep the error at each unit proportional to 
      #the contribution of that unit towards total 
      #error of the process.
      while(flag){
        flag <- FALSE
        yc = activte_fun(x,w,b)
        for (i in 1:nrow(x)){
          if (y[i] != yc[i]){
            w <- w + learn_rate * y[i] * x[i,]
            b <- b + learn_rate * y[i] * (Ecd_dist)^2
            err[i] <- err[i] + 1
            flag <- TRUE
          }
        }
        count = count + 1
        if(count > 1000)  #flag to break the loop after 1000 iterations                  
          break
      }
      s <- euclidean_dist(w)
      return(list(w= w/s, b = b/s, error = err, iteration = count))
    }
    
    #Perceptron function to test the model
    perceptron_Test <- function(x,w,b){
      
      yc = activte_fun(x,w,b)
      
      return(list(predtcd_value = yc))
    }
    
    #Training the model
    p_1 = perceptron(Train_Grp_1_data,Train_Grp_1_exptd)
    p_2 = perceptron(Train_Grp_2_data,Train_Grp_2_exptd)
    p_3 = perceptron(Train_Grp_3_data,Train_Grp_3_exptd)
    
    #Testing the model
    p_test_1 <- perceptron_Test(TestData[,(1:4)],p_1$w,p_1$b)
    p_test_2 <- perceptron_Test(TestData[,(1:4)],p_2$w,p_2$b)
    p_test_3 <- perceptron_Test(TestData[,(1:4)],p_3$w,p_3$b)
    
    #Combining the predicted values
    Predctd_owl_type <- function(Test_model_Grp_1,Test_model_Grp_2,Test_model_Grp_3){
      
      Predcted_Type <- vector(mode = "numeric", length = nrow(TestData))
      
      for(i in 1:length(TestData$Type)){
        if((p_test_1$predtcd_value[i] == 1) & (p_test_3$predtcd_value[i] == -1)){
          Predcted_Type[i] = "LongEaredOwl"
        }
        
        if((p_test_2$predtcd_value[i] == 1) & (p_test_1$predtcd_value[i] == -1)){
          Predcted_Type[i] = "SnowyOwl"
        }
        
        if((p_test_3$predtcd_value[i] == 1) & (p_test_2$predtcd_value[i]== -1)){
          Predcted_Type[i] = "BarnOwl"
        }
      }
      return(Predcted_Type)
    }
    
    Prcted_OWL_Type <- Predctd_owl_type(p_test_1,p_test_2,p_test_3)
    
    confusion_matrix <- table(TestData$Type,Prcted_OWL_Type)
    
    Accuracy <- ((confusion_matrix["BarnOwl","BarnOwl"]+
                    confusion_matrix["LongEaredOwl","LongEaredOwl"]+
                    confusion_matrix["SnowyOwl","SnowyOwl"])/length(TestData$Type))
    
    Accuracy_list[k] <- Accuracy
    
    #Display the output
    Output <- list(
      Actual_Owl_Type = TestData$Type,
      Predicted_Owl_Type = Prcted_OWL_Type,
      Confusion_Matrix = confusion_matrix,
      Current_itr_Accuracy = Accuracy,
      List_Of_Accuracies = Accuracy_list,
      Perceptron_1_weights = p_1$w,
      Perceptron_1_bias = p_1$b,
      Perceptron_2_weights = p_2$w,
      Perceptron_2_bias = p_2$b,
      Perceptron_3_weights = p_3$w,
      Perceptron_3_bias = p_3$b,
      Perceptron_1_Epochs = p_1$iteration,
      Perceptron_2_Epochs = p_2$iteration,
      Perceptron_3_Epochs = p_3$iteration
    )
    
    print(Output)
    
    print(ggplot()+
            geom_smooth(aes(x = c(1:250), y = p_1$err[1:250], colour = "LongEared vs SnowyOwl"), se = F)+
            geom_smooth(aes(x = c(1:250), y = p_2$err[1:250], colour = "SnowyOwl vs BarnOwl"), se=F)+
            geom_smooth(aes(x = c(1:250), y = p_3$err[1:250], colour = "BarnOwl vs LongearedOwl"), se=F)+
            ggtitle("Error vs Epochs")+
            xlab("Epochs")+
            ylab("Error")+
            theme(plot.title = element_text(hjust = 0.5)))
    
  }
 
  cat("Mean Accuracy is:", mean(Accuracy_list))

}

Classification_Algo(Data_Set,10)


  
