# Introduction

This is a homework specializing on experiment tracking with mlflow.
Here I created one docker container to run jupyter server and another container for mlflow server and managed these container with docker compose.

# Dataset
I just used famous iris dataset for this toy project.

# Prerequisites
In order to run the project, you should have git, docker, and docker compose plugin installed.

# Running the project
In order to run the project:
1. open git bash.
2. clone the repository with command:
```bash
git clone https://github.com/7r1s7on/MLE.git
```
3. ensure the terminal is located at "....../MLE/Module 3 Experiment Tracking/".
4. run command
```bash
docker compose up --build
```
After this docker will start pulling the jupyter image from the hub. When this finishes, it will start downloading necessary packages for project execution. When it is done, containers will start their executions.

When it finishes you can access mlflow UI for model tracking, to do so -> open your browser and paste `http:/localhost:5000`. You will the experiments with their runs there.

You can also access jupyter server and notebooks with: `http:/localhost:8888`.

This is a simple practice for experiment tracking, take with a grain of salt.