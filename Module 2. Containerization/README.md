# Module 2 - Containerization

## Introduction

This project is made after a kaggle competition. https://www.kaggle.com/competitions/playground-series-s4e3/overview.
The problem is multiclass classification of metal steel production defects.

There are 7 target columns. `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`.

All variables are in form of number. No NA values.

Around 32k samples.

## Prerequisite
In order to run the project, you should have git, docker.

## Running the project

In order to run the project:
1. open git bash.
2. clone the project into some folder with a command 
```bash 
git clone https://github.com/7r1s7on/MLE.git
```
3. ensure the terminal is located at "....../MLE/Module 2. Containerization/".
4. run command 
```bash
chmod +x run_project.sh
```
5. execute the script with 
```bash
./run_project.sh
```

After this, the project will start its execution, the resulting prediction.csv file will be located in `.../MLE/Module 2. Containerization` folder.