import random
import math
import matplotlib.pyplot as plt

#The two variables below stores the entire data and the corresponding labels
vector_of_data=[]
vector_of_labels=[]


#a function to parse the text file
#removes the \n and \t
#puts all the sentences into a list
def read_file_into_vector(filename):  
    file = open(filename, 'r+', encoding='utf-8')

    while True:
        
        lines = file.readlines()
        if not lines:
            break
        for line in lines:
            a=line.split("\t")
            vector_of_data.append(a[0])
            
            vector_of_labels.append(a[1].rstrip("\n"))
    return vector_of_data,vector_of_labels




#counts the number of examples that
#belongs to positive and the number of
#examples that belong to negative class
def count_class(label_vector):
    class_0_count,class_1_count=label_vector.count(0),label_vector.count(1)
    return class_0_count,class_1_count





#a function to put all the words in the document into one single vector
#also removes all the stopwords
#removes the duplicates also
def learn_to_vector(data_vector):
    
    stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such','only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    vector_of_words=[word for line in data_vector for word in line.split(" ") ]

    #https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    #used to remove duplicates
 #   vector_of_words=list( dict.fromkeys(vector_of_words ))
    return vector_of_words




#converts all the lines in the training data into bag of words representation.That is,it keeps count of each of the word occuring in sentence in a matrix of size of the entire vocabulary set
def vectorizer(data_vector,learned_words):
    vectorized_matrix=[]
    for line in data_vector:
        temp=[0 for i in range(len(learned_words))]
        for word in line.split(" "):
            if word in learned_words:
                index_of_word=learned_words.index(word)
                temp[index_of_word]+=1
            else:
                continue
        vectorized_matrix.append(temp)
    return vectorized_matrix




#splitting the entire data into train set and test set.80% goes into train and 20% into test set.
def split_train_and_test(vector_of_data,vector_of_labels):
    #train set 80%
    #test set 20%
    train_set=[]
    label_train_set=[]
    test_set=[]
    label_test_set=[]
    temp_indices=[]
    temp_train_set=random.sample(list(enumerate(vector_of_data)), math.floor(0.8*len(vector_of_data)))
    for index,value in temp_train_set:
        train_set.append(value)
        label_train_set.append(vector_of_labels[index])
        temp_indices.append(index)

    for i in range(len(vector_of_data)):
       if i not in temp_indices:
            test_set.append(vector_of_data[i])
            label_test_set.append(vector_of_labels[i])
    return train_set,test_set,label_train_set,label_test_set





#This function returns 4 values,that is the number of times a word occurs in each positive and negative class.Also,it returns the total number
#of words present in each positive class and negative class.
def count_word_given_class(vectorized_matrix,vector_of_labels):
 #   number_of_occurences_given_positive=[0 for i in range(len(vector_of_words))]
 #   number_of_occurences_given_negative=[0 for i in range(len(vector_of_words))]
    data_with_positive_label=[vectorized_matrix[i] for i in range(len(vector_of_labels)) if vector_of_labels[i]==1]
    data_with_negative_label=[vectorized_matrix[i] for i in range(len(vector_of_labels)) if vector_of_labels[i]==0]
    #https://stackoverflow.com/questions/14050824/add-sum-of-values-of-two-lists-into-new-list
    #use to add all lists inside a list element-wise
    number_of_word_occurences_given_positive=[sum(x) for x in zip(*data_with_positive_label)]
    number_of_word_occurences_given_negative=[sum(x) for x in zip(*data_with_negative_label)]
    m=n=0
    for i in number_of_word_occurences_given_positive:
        if i!=0:
            m+=i
    for i in number_of_word_occurences_given_negative:
        if i!=0:
            n+=i
    # m is number of tokens in positive class
    # n is number of tokens in negative class
    return number_of_word_occurences_given_positive,number_of_word_occurences_given_negative,m,n




#a function which is used for calculating maximum likelihood and predicts the label of the test data.It finally returns the accuracy of prediction on the test data.
def predict_using_max_likelihood(train_set,label_train_set,test_set,label_test_set):
    learned_words=learn_to_vector(train_set)
    vectorized_matrix=vectorizer(train_set,learned_words)
    number_of_word_occurences_given_positive,number_of_word_occurences_given_negative,m,n=count_word_given_class(vectorized_matrix,label_train_set)
    test_set=vectorizer(test_set,learned_words)
    #predicting on test set
    predicted_labels=[]
    class_0_count,class_1_count=count_class(label_train_set)
    
    for data_point in test_set:
        indices_where_word_is_present=[]
        likelihood_given_positive=0
        likelihood_given_negative=0
        
        for i in range(len(data_point)):
            if data_point[i]!=0:
                indices_where_word_is_present.append(i)
        for index in indices_where_word_is_present:
            if number_of_word_occurences_given_positive[index]!=0:
                likelihood_given_positive+=math.log(number_of_word_occurences_given_positive[index]/m)
                
            if number_of_word_occurences_given_negative[index]!=0:
                likelihood_given_negative+=math.log(number_of_word_occurences_given_negative[index]/n)
        likelihood_given_positive+=math.log(class_1_count/(class_0_count+class_1_count))
        likelihood_given_negative+=math.log(class_0_count/(class_0_count+class_1_count))
        if likelihood_given_positive>likelihood_given_negative:
                predicted_labels.append(0)
        elif likelihood_given_positive<likelihood_given_negative:
                predicted_labels.append(1)
        else:
            predicted_labels.append(random.sample([0,1],1))
            
    
    accurate_readings=0
   

    
    for i in range(len(label_test_set)):
        
        if (predicted_labels[i]==label_test_set[i]):
            accurate_readings+=1
    accuracy=accurate_readings/len(label_test_set)
    return accuracy

#a function which is used for calculating map and predicts the label of the test data.Here the m has to be given as a parameter.
#It finally returns the accuracy of prediction on the test data.
def predict_using_map(train_set,label_train_set,test_set,label_test_set,pseudo_count):
    learned_words=learn_to_vector(train_set)
    
    vectorized_matrix=vectorizer(train_set,learned_words)
    number_of_word_occurences_given_positive,number_of_word_occurences_given_negative,m,n=count_word_given_class(vectorized_matrix,label_train_set)
    test_set=vectorizer(test_set,learned_words)
    #predicting on test set
    predicted_labels=[]
    class_0_count,class_1_count=count_class(label_train_set)

    for data_point in test_set:
        indices_where_word_is_present=[]
        likelihood_given_positive=0
        likelihood_given_negative=0
        for i in range(len(data_point)):
            if data_point[i]!=0:
                indices_where_word_is_present.append(i)
        for index in indices_where_word_is_present:
             if number_of_word_occurences_given_positive[index]!=0:
                    likelihood_given_positive+=math.log((number_of_word_occurences_given_positive[index]+pseudo_count)/(m+pseudo_count*len(learned_words)))
             if number_of_word_occurences_given_negative[index]!=0:
                    likelihood_given_negative+=math.log((number_of_word_occurences_given_negative[index]+pseudo_count)/(n+pseudo_count*len(learned_words)))
        likelihood_given_positive+=math.log(class_1_count/(class_0_count+class_1_count))
        likelihood_given_negative+=math.log(class_0_count/(class_0_count+class_1_count))
        if likelihood_given_positive>likelihood_given_negative:
                predicted_labels.append(0)
        elif likelihood_given_positive<likelihood_given_negative:
                predicted_labels.append(1)
        else:
            predicted_labels.append(random.sample([0,1],1))
    
      
    accurate_readings=0
   

    
    for i in range(len(label_test_set)):
        
        if (predicted_labels[i]==label_test_set[i]):
            accurate_readings+=1
    accuracy=accurate_readings/len(label_test_set)
    return accuracy
       



#This fucntion is used to divide the data into 10 partitions and also to introduce the randomness along with it.
#Also the sampling has been done in a stratified fashion here.
def split_into_stratified_samples(train_set,label_train_set):
    #first do stratified sampling and seperate into 10 sets for 10-fold cv
    negative_samples=[]
    positive_samples=[]
    for i in range(len(label_train_set)):
        if label_train_set[i]=="0":
            negative_samples.append(train_set[i])
        else:
            positive_samples.append(train_set[i])
    random.shuffle(negative_samples)
    random.shuffle(positive_samples)
    split_size_positive=math.floor(len(positive_samples)/10)
    split_size_negative=math.floor(len(negative_samples)/10)
    sample_set_1_positive=positive_samples[split_size_positive*0:split_size_positive*1]
    label_set_1_positive=[1 for i in range(len(sample_set_1_positive))]
    sample_set_2_positive=positive_samples[split_size_positive*1:split_size_positive*2]
    label_set_2_positive=[1 for i in range(len(sample_set_2_positive))]
    sample_set_3_positive=positive_samples[split_size_positive*2:split_size_positive*3]
    label_set_3_positive=[1 for i in range(len(sample_set_3_positive))]                
    sample_set_4_positive=positive_samples[split_size_positive*3:split_size_positive*4]
    label_set_4_positive=[1 for i in range(len(sample_set_4_positive))]
    sample_set_5_positive=positive_samples[split_size_positive*4:split_size_positive*5]
    label_set_5_positive=[1 for i in range(len(sample_set_5_positive))]
    sample_set_6_positive=positive_samples[split_size_positive*5:split_size_positive*6]
    label_set_6_positive=[1 for i in range(len(sample_set_6_positive))]
    sample_set_7_positive=positive_samples[split_size_positive*6:split_size_positive*7]
    label_set_7_positive=[1 for i in range(len(sample_set_7_positive))]
    sample_set_8_positive=positive_samples[split_size_positive*7:split_size_positive*8]
    label_set_8_positive=[1 for i in range(len(sample_set_8_positive))]
    sample_set_9_positive=positive_samples[split_size_positive*8:split_size_positive*9]
    label_set_9_positive=[1 for i in range(len(sample_set_9_positive))]
    sample_set_10_positive=positive_samples[split_size_positive*9:split_size_positive*10]
    label_set_10_positive=[1 for i in range(len(sample_set_10_positive))]
    sample_set_1_negative=negative_samples[split_size_negative*0:split_size_negative*1]
    label_set_1_negative=[0 for i in range(len(sample_set_1_negative))]
    sample_set_2_negative=negative_samples[split_size_negative*1:split_size_negative*2]
    label_set_2_negative=[0 for i in range(len(sample_set_2_negative))]
    sample_set_3_negative=negative_samples[split_size_negative*2:split_size_negative*3]
    label_set_3_negative=[0 for i in range(len(sample_set_3_negative))]                
    sample_set_4_negative=negative_samples[split_size_negative*3:split_size_negative*4]
    label_set_4_negative=[0 for i in range(len(sample_set_4_negative))]
    sample_set_5_negative=negative_samples[split_size_negative*4:split_size_negative*5]
    label_set_5_negative=[0 for i in range(len(sample_set_5_negative))]
    sample_set_6_negative=negative_samples[split_size_negative*5:split_size_negative*6]
    label_set_6_negative=[0 for i in range(len(sample_set_6_negative))]
    sample_set_7_negative=negative_samples[split_size_negative*6:split_size_negative*7]
    label_set_7_negative=[0 for i in range(len(sample_set_7_negative))]
    sample_set_8_negative=negative_samples[split_size_negative*7:split_size_negative*8]
    label_set_8_negative=[0 for i in range(len(sample_set_8_negative))]
    sample_set_9_negative=negative_samples[split_size_negative*8:split_size_negative*9]
    label_set_9_negative=[0 for i in range(len(sample_set_9_negative))]
    sample_set_10_negative=negative_samples[split_size_negative*9:split_size_negative*10]
    label_set_10_negative=[0 for i in range(len(sample_set_10_negative))]
    sample_set_1=sample_set_1_positive+sample_set_1_negative
    sample_set_2=sample_set_2_positive+sample_set_2_negative
    sample_set_3=sample_set_3_positive+sample_set_3_negative
    sample_set_4=sample_set_4_positive+sample_set_4_negative
    sample_set_5=sample_set_5_positive+sample_set_5_negative
    sample_set_6=sample_set_6_positive+sample_set_6_negative
    sample_set_7=sample_set_7_positive+sample_set_7_negative
    sample_set_8=sample_set_8_positive+sample_set_8_negative
    sample_set_9=sample_set_9_positive+sample_set_9_negative
    sample_set_10=sample_set_10_positive+sample_set_10_negative
    label_set_1=label_set_1_positive+label_set_1_negative
    label_set_2=label_set_2_positive+label_set_2_negative
    label_set_3=label_set_3_positive+label_set_3_negative
    label_set_4=label_set_4_positive+label_set_4_negative
    label_set_5=label_set_5_positive+label_set_5_negative
    label_set_6=label_set_6_positive+label_set_6_negative
    label_set_7=label_set_7_positive+label_set_7_negative
    label_set_8=label_set_8_positive+label_set_8_negative
    label_set_9=label_set_9_positive+label_set_9_negative
    label_set_10=label_set_10_positive+label_set_10_negative
    return sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10





#Joining the partitioned data into different folds for the purpose of cross-validation has been performed here.
def split_into_cross_validation(sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10):
    sample_sets=[sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10]
    label_sets=[label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10]
    data_folds=[]
    label_folds=[]
    
    for i in range(0,10):
        temp_fold=[]
        temp_label_fold=[]
        for j in range(0,10):
            if j!=i:
                temp_fold+=sample_sets[j]
                temp_label_fold+=label_sets[j]
            else:
                continue
        data_folds+=[temp_fold]
        label_folds+=[temp_label_fold]
    return data_folds,label_folds
    


            
##experiment 1
##The following code is for the 1st experiment
def experiment_1_plot_accuracy_with_data_size(sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10,test_set,new_test_labels,m):
    accuracies=[]
    #size 0.1N
    
    train_set=sample_set_1
    label_train_set=label_set_1
    if m==0:
        accuracy_cv_1N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_1N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_1N*100]

    #size 0.2N

    train_set=sample_set_1+sample_set_2
    label_train_set=label_set_1+label_set_2
    if m==0:
        accuracy_cv_2N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_2N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_2N*100]

    #size 0.3N

    train_set=sample_set_1+sample_set_2+sample_set_3
    label_train_set=label_set_1+label_set_2+label_set_3
    if m==0:
        accuracy_cv_3N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_3N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_3N*100]

    #size 0.4N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4
    if m==0:
        accuracy_cv_4N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_4N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_4N*100]

    #size 0.5N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4+sample_set_5
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4+label_set_5
    if m==0:
        accuracy_cv_5N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_5N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_5N*100]

    #size 0.6N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4+sample_set_5+sample_set_6
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4+label_set_5+label_set_6
    if m==0:
        accuracy_cv_6N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_6N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_6N*100]

    #size 0.7N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4+sample_set_5+sample_set_6+sample_set_7
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4+label_set_5+label_set_6+label_set_7
    if m==0:
        accuracy_cv_7N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_7N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_7N*100]

    #size 0.8N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4+sample_set_5+sample_set_6+sample_set_7+sample_set_8
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4+label_set_5+label_set_6+label_set_7+label_set_8
    if m==0:
        accuracy_cv_8N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_8N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_8N*100]

    #size 0.9N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4+sample_set_5+sample_set_6+sample_set_7+sample_set_8+sample_set_9
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4+label_set_5+label_set_6+label_set_7+label_set_8+label_set_9
    if m==0:
        accuracy_cv_9N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_9N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_9N*100]
    #size 1.0N

    train_set=sample_set_1+sample_set_2+sample_set_3+sample_set_4+sample_set_5+sample_set_6+sample_set_7+sample_set_8+sample_set_9+sample_set_10
    label_train_set=label_set_1+label_set_2+label_set_3+label_set_4+label_set_5+label_set_6+label_set_7+label_set_8+label_set_9+label_set_10
    if m==0:
        accuracy_cv_10N=predict_using_max_likelihood(train_set,label_train_set,test_set,new_test_labels)
    else:
        accuracy_cv_10N=predict_using_map(train_set,label_train_set,test_set,new_test_labels,m)
    accuracies+=[accuracy_cv_10N*100]
    return accuracies
    




##experiment 2
##The following function is used to perform the 2nd experiment which has been specified.
def experiment_2_plot_cvaccuracy_with_m(data_folds,label_folds,sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10,test_set,new_test_labels,m):
           
           accuracy_cv_1=predict_using_map(data_folds[0],label_folds[0],sample_set_1,label_set_1,m)
           accuracy_cv_2=predict_using_map(data_folds[1],label_folds[1],sample_set_2,label_set_2,m)
           accuracy_cv_3=predict_using_map(data_folds[2],label_folds[1],sample_set_3,label_set_3,m)
           accuracy_cv_4=predict_using_map(data_folds[3],label_folds[1],sample_set_4,label_set_4,m)
           accuracy_cv_5=predict_using_map(data_folds[4],label_folds[1],sample_set_5,label_set_5,m)
           accuracy_cv_6=predict_using_map(data_folds[5],label_folds[1],sample_set_6,label_set_6,m)
           accuracy_cv_7=predict_using_map(data_folds[6],label_folds[1],sample_set_7,label_set_7,m)
           accuracy_cv_8=predict_using_map(data_folds[7],label_folds[1],sample_set_8,label_set_8,m)
           accuracy_cv_9=predict_using_map(data_folds[8],label_folds[1],sample_set_9,label_set_9,m)
           accuracy_cv_10=predict_using_map(data_folds[9],label_folds[1],sample_set_10,label_set_10,m)
           accuracy=accuracy_cv_1+accuracy_cv_2+accuracy_cv_3+accuracy_cv_4+accuracy_cv_5+accuracy_cv_6+accuracy_cv_7+accuracy_cv_8+accuracy_cv_9+accuracy_cv_10
           avg_accuracy=accuracy/10
           return avg_accuracy*100    




def main_function(data):
    vector_of_data,vector_of_labels=read_file_into_vector(data)
    train_set,test_set,label_train_set,label_test_set=split_train_and_test(vector_of_data,vector_of_labels)
    new_test_labels=[]
    for i in label_test_set:
        if i=="1":
            new_test_labels+=[1]
        else:
            new_test_labels+=[0]
            
    sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10=split_into_stratified_samples(vector_of_data,vector_of_labels)
    data_folds,label_folds=split_into_cross_validation(sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10)        

    print("Running experiment 1")
    accuracy_for_m_0=experiment_1_plot_accuracy_with_data_size(sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10,test_set,new_test_labels,0)               
    accuracy_for_m_1=experiment_1_plot_accuracy_with_data_size(sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10,test_set,new_test_labels,1)               
    size=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    plt.figure(1)
    plt.plot(size,accuracy_for_m_0,label="m=0")
    plt.plot(size,accuracy_for_m_1,label="m=1")
    plt.legend("M-value")
    plt.xlabel("Size of data")
    plt.ylabel("Accuracy")
    plt.title("'Variation of Accuracy with size of data")
    plt.grid()
    error_table_0=[]
    for i in accuracy_for_m_0:
        error_table_0+=[i-sum(accuracy_for_m_0)/10]
    error_table_1=[]
    for i in accuracy_for_m_1:
        error_table_1+=[i-sum(accuracy_for_m_1)/10]
    plt.figure(2)
    plt.errorbar(size, error_table_0, marker='s', mfc='red',mec='green', ms=20, mew=4,label="m=0")
    plt.errorbar(size, error_table_1, marker='s', mfc='red',mec='green', ms=20, mew=4,label="m=1")
    plt.legend("M-value")
    plt.xlabel("Size of data")
    plt.ylabel("Error")
    plt.title("'Variation of Error with size of data")
    plt.grid()
    print("Done with experiment 1")
    print("Running Experiment 2")
    m_values=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
    accuracy_m=[]
    for m in m_values:
        print("Running for m-value",m)
        accuracy_m+=[experiment_2_plot_cvaccuracy_with_m(data_folds,label_folds,sample_set_1,sample_set_2,sample_set_3,sample_set_4,sample_set_5,sample_set_6,sample_set_7,sample_set_8,sample_set_9,sample_set_10,label_set_1,label_set_2,label_set_3,label_set_4,label_set_5,label_set_6,label_set_7,label_set_8,label_set_9,label_set_10,test_set,new_test_labels,m)]
            
    print(accuracy_m)
    error_table=[]
    for i in accuracy_m:
            error_table+=[i-sum(accuracy_m)/10]

    plt.figure(3)
    plt.errorbar(m_values, error_table, marker='s', mfc='red',mec='green', ms=20, mew=4)
    plt.legend()
    plt.xlabel("Values of m")
    plt.ylabel("Error")
    plt.title("Variation of Cross Validation Error with m")
    plt.grid()

    plt.figure(4)
    plt.plot(m_values,accuracy_m)
    plt.scatter(m_values,accuracy_m)
    plt.legend()
    plt.xlabel("Values of m")
    plt.ylabel("Average accuracy after K-fold validation")
    plt.title("Variation of average cross-validation accuracy with m")
    plt.grid()
    plt.show()        
    print("Done with experiment 2")
                
                
data=input("Enter the filename of the dataset stored in the same folder.The extension of the file also has to be entered :" )
main_function(data)
        
        

