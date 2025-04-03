# Module 2 - Conteinerization

## Introduction

This project is made after a kaggle competition. https://www.kaggle.com/competitions/playground-series-s4e3/overview.
The problem is multiclass classification of metal steel production defects.

There are 7 target columns. `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`.

All variables are in form of number. No NA values.

Around 32k samples.

In order to run the project:
1. clone the repository.
2. run data_processing.py script. It will create processed_train.csv.
3. run train.py.
4. run inference.py.
5. build docker train image with `docker build --target trainer -t tabnet-trainer .`. This will create a train image.
6. execute the `docker run   -v $(pwd)/data:/app/data   -v $(pwd)/m   -v $(pwd)/logs:/app/logs   tabnet-trainer` command to run the container.
7. build docker inference image with `docker build -t tabnet-inference .`. This will create inference image.
8. execute `docker run -v "/$(pwd -W)/models:/app/models"            -v "/$(pwd -W)/data:/app/data"            -v "/$(pwd -W)/output:/app/output"            tabnet-inference` command to run the inference container.
