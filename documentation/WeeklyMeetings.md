# Q2 - Avocado Price Prediction
 
## Meeting 09.07.2020

### Questions
* The Data   
   *   

* The Model    
   * 

* The Project   
   * Any questions on git or GitLab? 

### Notes   
*

### Goals
* ARIMA model
* SARIMA model

* RNN (Recurrent Neural Network) and LSTN

* * *

## Meeting 02.07.2020

### Questions

### Notes   
*

### Goals
* Data Analysis and shallow models

* * *

## Meeting 26.06.20

### Questions
* The Data
   * Recommendation: Project code on GitLab + local or directly in GDrive   
    -> https://stackoverflow.com/questions/48350226/methods-for-using-git-with-google-colab/48370584   
    -> Anaconda
   * Should we consider subsetting representatives? How do we do that?   
    -> whole dataset / no subsetting
   * Preprocessing necessary? deleting NAs?   
    -> look at irregularities/ look up weather data from other datasets   
    -> new dataset by Hass Avocado   
    -> Null values/ mean value/ duplicate check/ key integretiy   
    -> Maybe a tutorial?!    
    -> Kernels on kaggle dataset website
   * Which of the data (features) in the excel sheet will be fed into the model?   
    -> for autoregression leave out irrelevant features   
    -> generate new feature for linear regression for example season   
    -> feed all to dl model   
   * For the paper: Present dataset with graphs, etc (=DataAnalysis) or just describe based on metadata?  
    -> using notebooks and commenting -> use it for the paper   
    -> paper topic is more about comparison
   * For the paper: Maximum amount of pages/ words?   
    -> min 4 pages   
   * Difference Colab demo and code? It is basically the same just with text?! What is it about?
    -> use local and then upload to google drive
   * License file: Of our code? ()= GNU General Public License) Or imported libraries? Or used dataset?    
    -> our code/ project   
	-> for paper: referencing/ sources/ etc of datasets etc     

* The Model    
   * neural network to perform a regression function
   * The tutorial I followed uses a module called TFANN which stands for “TensorFlow Artificial Neural Network.” - are we allowed to use frameworks like that? or shall we build it from scratch? they probably told us at some point but i don't remember    
-> shallow model compared with DL model (maybe with regularization)

* The Project   
   * Should we have a timetable for the project? Due when should we accomplish what?   
    -> internal structure for ourselves   
    -> fast sprint (1 week data analysis with plots = questions for dataset)   
    -> baseline model for autoregression/ DL model (training methods = general to adjust for other models)
   * Should we work on everything together or divide tasks?   
    -> collaborate first model   
    -> then split up do analysis separately and then collect all results

### Notes   
Highly robust models might utilize external data such as news, time of the year, social media sentiment, weather, price of competitors, market volatility, market indices, etc 

### Goals
* 1 week: Data Analysis and baseline model   
* maybe look for second dataset > merge NOT NECESSARY THO