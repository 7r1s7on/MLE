import os
import pandas as pd

from .constants import DATA_PATH

def generate_data(data_path):
    """Manually splitting dataset into two parts based human factors and educational factors"""
    data = pd.read_csv(data_path)
    human_factor_columns = "sex age address famsize Pstatus Mjob Fjob guardian famsup internet romantic freetime goout health".split()
    edu_factor_columns = (
        "Medu Fedu reason studytime failures schoolsup absences G1 G2 G3".split()
    )
    human_factor_data = data[human_factor_columns]
    edu_factor_data = data[edu_factor_columns]
    human_factor_data.to_csv(os.path.join(DATA_PATH, "raw/human_factor_data.csv"))
    edu_factor_data.to_csv(os.path.join(DATA_PATH, "raw/edu_factor_data.csv"))
    print("Data generated successfully")
