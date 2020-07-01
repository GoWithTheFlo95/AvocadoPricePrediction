# Q2 - Avocado Price Prediction

Create a deep learning model, which predicts future avocado prices based on historical price data. In which regions does your model perform best? How does your model compare to standard methods, such as autoregression?

Standard models: https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/   
Dataset: https://www.kaggle.com/neuromusic/avocado-prices

* * *

## General Information

* Deliverables are: **project code** + **colab demo** + **executive summary** (.pdf)    
* **Project submission deadline**: 23rd of July, 23:59 UTC+2 (Berlin)
* **Summary submission deadline**: 30th of July, 23:59 UTC+2 (Berlin)

+ Zip your submission files as: dl_’projectID’.zip
+ Send your submission to machinelearning.dhc@gmail.com    

- MSc.: Code makes up 30% of your project grade, project summary 70%   
- We consider **completeness**, **quality** and **style of all deliverables** 

### Project structure

File                    | Description
----------------------- | ----------------------------------------------
models/modelX.py        | trained model files (deepLearning + Autoregression model)
data_loading.py 	    | preprocessing, data iterators
architecture.py 	    | model architecture class(es)
train.py 		        | instantiates a model, conducts the training and saves the model
test.py 			    | evaluation metric functions
demo.colab              |
summary.pdf             |
README		            | Describe clearly how to run your code
LICENSE		            | Under which license do you distribute your project?

### Code

- in Pytorch or tensorflow/keras
- Comment and structure your code in functions and classes to improve readability and understanding
- Consider using argparse to specify parameters
- Loading, preprocessing, training and evaluation are project-dependent, explain your choices

### Colab Demo

- Colab demo is a selected subset of your code and shall quickly demonstrate your project and give insights into your results
- Involves:
    * data loading
    * exemplary visualization
    * exemplary model inference
    * output presentation
- Demo colab file should have been executed before sharing / no need to re-run should be required
- Extent: min 4 pages (MSc.) excl. references, exported as pdf file
- We use LaTeX and Word templates from NIPS 2015: https://nips.cc/Conferences/2015/PaperInformation/StyleFiles

### Executive Summary

Structure | example questions
--- | ---
Project summary | What is your project about? What are its goals?
Related Work | Have others approached what you did? Which works are related to yours?
Dataset | What are characteristics of your dataset, e.g. size, input/target output, dimensions, conducted preprocessing, dataset splits
Architecture & Training | Which machine learning architecture have you chosen? How have you trained your model? Which experiments have you run? How did you select hyperparameters?       
Evaluation | Which evaluation strategy and metrics have you chosen? How does your model perform?
Discussion | Is your model performing well? How could your model be improved? Which challenges were involved? What is future work? Effects on different hyperparameters (epochsize, learning rate) on model -> https://medium.com/pytorch/using-optuna-to-optimize-pytorch-hyperparameters-990607385e36 

- When writing the summary, always explain the ‘why’ after you specified the ‘what’.
- Use references if required

### Google collaboration

- In a notebook, select Runtime > Change Runtime type > GPU
- Get access to your python modules within colab:
   * Write your code in a .py file, upload it to Drive   
   * Mount your drive, authenticate yourself   
        ```
        from google.colab import drive
        drive.mount('/content/drive')
        ```
   * Append the location of your .py files to the path
        ```
        import sys
        sys.path.append('/content/drive/My drive/Colab Notebooks/')
        ```
   * Import [3] and run your .py files [4]
        ```
        from skriptName import funcName
        funcName()
        ```

* Data Access  
  
        from google.colab import drive
        drive.mount('/content/drive')

        import matplotlib.pylot as plt
        from PIL import Image 

        !ls
        %cd drive/My\ Drive/Colab\ Notebooks

        img = Image.open("test_img.png")
        plt.imshow(img)
 