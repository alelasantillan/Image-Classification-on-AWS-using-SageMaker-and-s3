# Image Classification on AWS using SageMaker and s3
## Udacity Project 3: AWS Machine Learning Engineer Nanodegree Program (2021-2022)

The present work consists in taking a large datafile of dog images classified by breed and construct
a model to be able to predict the breeds of dogs on the test file.

For that main task several minor steps are to be executed:
1. Download the data from a public available repo using wget and unzip data into the AWS filesystem (s3)
2. Train the last layer of a pretrained model (Resnet18) usign the image the data on s# to perfrom a hyperparameter optimization.
3. Use the hyperparameters obtained in 3. to retrain the same model using the Profiling and Debugging of the model  parameters using hooks, a module avaible from SageMaker, to avoid *automagically vanishing gradients, overfitting, overtraining, poor weight initialization and generate a Profiler Report.
4. Deploy the model as an endpoint to perform inference on the breed of selected images. 


## Step 0: Project Set Up and Installation
You must be logged to a AWS account and search for SageMaker to get into the SageMaker console. 
Launch Sagemaker Studio.
It could take a few minutes to start. 
Once open you will be able to upload all the files of this project and run the Jupyter Notebook that comprises the whole process described above.


The files to upload are: 
train_adn_deploy.ipynb, running each cell will execute the whole process and three auxiliary python scripts that are called by the notebook:
hpo.py, contains the prediction model, the training loop as well as the validation and testing tasks in step 2 for hyper parameter optimization.
train_model.py, esentially identical to hpo.py, but with the hooks of SageMaker module that performs debugging of the model in step 3.
endpoint_inference.py responsible for invoking the endpoint created by the notebook and return the prediction.

Open the Jupyter Notebook and select the kernel as follows:
<br/>
<img src="images/kernel.png" width="50%">
<br/>
<br/>
and the instance:<br/>
<img src="images/instance.png" width="50%">
<br/>




## Step 1: Dataset
The first step is to get data from the web, unzip it, and make it available into s3.
You must generate an s3 bucket and provide the bucket name to the corresponding cell.
Also you must use the region global for the s3 bucket, but the region US-east-1 (N.Virginia) for the SageMaker studio.
The dataset can be downloaded [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

### Dependencies

```
Python 3.7
Pytorch AWS Instance
```
## Files Used in the notebook

- `hpo.py` - This script file contains code that will be used by the hyperparameter tuning jobs to train and test/validate the models with different hyperparameters to find the best hyperparameter
- `train_model.py` - This script file contains the code that will be used by the training job to train and test/validate the model with the best hyperparameters that we got from hyperparameter tuning
- `endpoint_inference.py` - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.
- `train_and_deploy.ipynb` - This jupyter notebook contains all the code and the steps performed in this project and their outputs.

## Hyperparameter Tuning
- The ResNet model represents the deep Residual Learning Framework to ease the training process.
- A pair of fully connected Neural Networks has been added on top of the pretrained model to perform the classification task with 133 output nodes.
- AdamW from torch.optm is used as an optimizer.
- The Following hyperparamets are used:
    - Learning rate-  0.01x to 100x
    - eps -  1e-09 to 1e-08
    - Weight decay -  0.1x to 10x
    - Batch size -  [ 64, 128 ]

The `hpo.py` script is used to perform hyperparameter tuning.

![Hyperparameters Tuning](Snapshots/Hyperparameter_Tuning_Job.png "Hyperparameters Tuning") ![Hyperparameters](Snapshots/Hyperparameters.png "Hyperparameters")

###Training Jobs
![Training Jobs](Snapshots/Training%20Jobs.png "Training Jobs")

## Debugging and Profiling
The Graphical representation of the Cross Entropy Loss is shown below.
![Cross Entropy Loss](Snapshots/Profiling%20and%20Debugging.png "Cross Entropy Loss")

Is there some anomalous behaviour in your debugging output? If so, what is the error and how will you fix it?
- There is no smooth output line and there are different highs and lows for the batch sets.
  If not, suppose there was an error. What would that error look like and how would you have fixed it?
- A proper mix of the batches with shuffling could help the model learn better
- Trying out different neural network architecture.

### Profiler Output
The profiler report can be found [here](profiler_report/profiler-output/profiler-report.html).

![Events](Snapshots/Events%20Logging.png "Events Bridge")
## Model Deployment
- Model was deployed to a "ml.t2.medium" instance type and "endpoint_inference.py" script is used to setup and deploy our working endpoint.
- For testing purposes ,few test images are stored in the "testImages" folder.
- Those images are fed to the endpoint for inference/
- The inference is performed using both the approaches. 
    1. Using the Predictor Object 
    2. Using the boto3 client.
  
![End Point Deployment](Snapshots/End%20Point.png "End Point")

