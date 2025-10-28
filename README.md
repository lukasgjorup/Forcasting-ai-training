# Forcasting-ai-training
A repo for training the ai model for forcasting


## Set up for Windows
* winget install AstralSoftware.Uv //for installing uv use in powershell
* uv sync // using in project folder to sync dependecies and python version. this will create a .venv for the project, that you should use.


## adding dependencies
add the dependency to pyproject.toml
then run uv sync in terminal

## downloading dataset
first download the dataset, this can be done with curl or from this [website](https://www.kaggle.com/api/v1/datasets/download/jeanmidev/smart-meters-in-london).
```shell
curl -L -o data/smart_meters_in_london.zip https://www.kaggle.com/api/v1/datasets/download/jeanmidev/smart-meters-in-london
```
Then unzip the file so that the files are in the data directory.
