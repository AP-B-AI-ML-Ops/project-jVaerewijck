# Describe your project
# Table of Contents

1. [Datasets](#datasets)
2. [Project Explanation](##project-explanation)
3. [Flows & Actions](#flows-&-actions)
4. [Training](#training)
5. [Predicting](#predicting)
## Datasets
I will primarily be using the Homicide Reports, 1980-2014 dataset. I can also get extra data from the Homicides database. Both of these are on kaggle.
## Project Explanation
The goal is to predict what kind of people are most likely going to be murdered next as well as the perpetrator
## Flows & Actions
There will be a flow to load the data, to preprocess the data and one to train the model/ test the model.
In the first flow there will be seperate tasks to load the data and to save the data.  
The second flow will have a task to load the data, a task to preprocess the data and a task to store the changed data.  
The train flow will have a task to load the data, to train the model and to test the model, there would also be one to save the model.  
## Training
When in the project directory simply type and execute "python main.py" in the console to start the training.  
The number of epochs starts extremely low, this is to reduce the training time to a managable timeframe.
The same is true for trials.  
Inside the training there are several checks to stop training early, these can be changed but training times will increase dramatically.  
There is a change the collect flow will ask you to make a new file called kaggle.json in a specific location, just follow the instructions, this part cant be pushed to github.
## Predicting
You should be able to use predict.py in the project folder to predict.  
The file should also be executed in the project folder.  
The run_id is printed in the console at the end of the flow, this has to be manualy entered at the top of the file.  
