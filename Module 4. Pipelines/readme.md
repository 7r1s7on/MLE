# Moduel 4 - Pipelines

## Introduction

This is a howework specialized on build pipelines with the help of Apache Airflow.

## Dataset

The dataset used for this homework is taken from kaggle: https://www.kaggle.com/datasets/devansodariya/student-performance-data/data

The dataset is about student performance. It consists of 395 samples and 33 columns, some of them are: 

* school ID
* gender
* age
* size of family
* Father education
* Mother education
* Occupation of Father and Mother
* Family Relation
* Health
* Grades

## Prerequisites

In order to run the project, you should have git and docker installed. Also the kaggle account is required for dataset download.

## Running the project

In order to run the project:

1. open git bash
2. clone the repository
    ```bash
    git clone https://github.com/7r1s7on/MLE.git
    ```
3. ensure the terminal is located at "....../MLE/Module 4. Pipelines/".
4. run command
    ```bash
    docker build -t airflow_image .
    ```
    *this will build the docker image.*

5. in order to run the container, execute the following command
    ```bash
    docker run -d -p 8080:8080 \
    -e KAGGLE_USERNAME="YOUR_KAGGLE_USERNAME" \
    -e KAGGLE_KEY="YOUR_KAGGLE_KEY"\
    --name airflow_container\
    airflow_image
    ```
    **Make sure to replace with your kaggle credentials.** There is insturction how to get username and key from kaggle below.

After this, wait ~30 seconds, then open your browser and paste the following link:
http://localhost:8080

Apache Airflow login page will be opened, enter with username and password - **admin**.

The Airflow home page will be opened. There will be one line of DAG, on the right side of that line there is 'play' button, this will start the execution. After some time, execution will end and results shown.

## Instruction to get username and key from kaggle

After signing/logging into kaggle, click on your profile icon in the top-right corner -> click on **settings** -> scroll down till **API** header is shown -> click on **Create New Token**, the kaggle.json file will be downloaded -> open it with any IDE, there will be displayed your username and key.