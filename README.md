# FASTAI LIVE VIDEO LOGO OBFUSCATION
The Project aims To DETECT and OBFUSCATE specific brand logos from clothing in a live video feed. 

## Problem Statement
A lot of movies, TV shows, or music videos blur out, remove or cover out the logos and/or brand names of certain companies when they appear on screen. This is known as product displacement. 

Whenever they are including the product or service of a particular brand in their creative work, they are in one way advertising the product of that brand amongst its viewers.  Many producers often expect the brands to pay for such advertisement. But, when the brands refuse to sponsor or pay for such inclusion, the producers often blur out or remove the logo of the brand from its products. 

While it is certainly not illegal to use trademarked products in the visual media by films, TV shows, and music videos, they often resort to product displacement, for a number of reasons, ranging from ‘avoiding the legal battles’ to ‘not wanting to annoy a sponsor’.

![Project workflow](docs/images/project_workflow.png)


## AWS Serverless Application Model
The AWS Serverless Application Model (SAM) is an open-source framework for building serverless applications. It provides shorthand syntax to express functions, APIs, databases, and event source mappings.
https://aws.amazon.com/serverless/sam/

## Run your own Nike Obfuscator 

*Dataset*: To train and peform inference on our models we first need data for model training. 

### Environment 
We will use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) 
to create our environment. Create a python 3.8 envrionment first: 

``` conda create -n py38 python=3.8 ```

Activate the environment: 

``` conda activate py38 ```

Then install dependencies: 

``` pip3 install -r requirements.txt ```

## Training your own model 
Run `train.py` (im guessing we'll have a training script?). This will generate a model saved to a specified directory. 

``` python3 train.py --data-dir --model-save-dir ```

## Inference
We can run inference on models in this repository, or your own trained model: 

``` python3 infer.py --model_dir [] --image [] ```

## Webcam client 
Simply run `webcam_client.py` to run obfuscation live. 

``` python3 webcam_client.py ```

Adjustments can be made to the running client to toggle bounding boxes, switch out models and change the 
frame rate of the stream. 

_insert a screenshot here !_

We have only trained our provided models with Nike logos for the time being. 
So make sure you have a Nike logo ready!

## End-to-end notebook
If you're a more interactive person, or you like experimenting, see __ in notebooks 
to view the training and inference code end-to-end.