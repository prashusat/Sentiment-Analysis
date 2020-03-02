
Overview:
In this assignment you will implement the Naive Bayes algorithm with maximum likelihood and
MAP solutions and evaluate it using cross validation on the task of sentiment analysis (as in
identifying positive/negative product reviews).
Text Data for Sentiment Analysis:
We will be using the Sentiment Labelled Sentences Data Set"1 that includes sentences labelled with
sentiment (1 for positive and 0 for negative) extracted from three domains imdb.com, amazon.com,
yelp.com. These form 3 datasets for the assignment.
Each dataset is given in a single le, where each example is in one line of that le. Each such
example is given as a list of space separated words, followed by a tab character (\t), followed by
the label, and then by a newline (\n). Here is an example from the yelp dataset:
Crust is not good. 0
The data, which is hosted by the UCI machine learning repository, is linked through the course
web page.



Instruction to run the code

1) The code has been split into different functions and all of it is inside the same .py file.

2) Running the code is a pretty simple process here.In order to run it from the terminal, we have to type the following:
    
            “python3 machinelearning.py”
  
3)Next it will ask you for the filename of the text file that you want to         
        input and we have to enter the name along with the extension.
  
4)Now, the code should start running.It will keep on prompting on the 
         terminal when its done with experiment-1 and as it runs experiment-2
         on different values of m as follows.
  
5)The program will take a little over a minute to run for each dataset
          and then it shall terminate with 4 different graphs as output.
