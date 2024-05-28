import os
import pickle
import pandas as pd
import zipfile
from prefect import flow, task
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim


@task
def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


@flow
def unzip_file(data_path: str):
    zip_file_path = os.path.join(data_path, "database.csv.zip")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)


@task
def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    df.replace({"": pd.NA, " ": pd.NA}, inplace=True)
    df = df.dropna()
    # geolocator = Nominatim(user_agent="myGeocoder")
    # def get_coordinates(city, state):
    #     location = geolocator.geocode(city + ', ' + state)
    #     if location:
    #         return location.latitude, location.longitude
    #     else:
    #         return None, None
    # df['latitude'], df['longitude'] = zip(*df.apply(lambda row: get_coordinates(row['City'], row['State']), axis=1))
    df = df.drop(
        columns=[
            "Agency Code",
            "Agency Name",
            "Agency Type",
            "City",
            "State",
            "Year",
            "Month",
            "Incident",
            "Crime Type",
            "Victim Count",
            "Perpetrator Count",
            "Record Source",
        ]
    )

    return df


@task
def preprocess(df: pd.DataFrame):
    le_crime_solved = LabelEncoder()
    le_victim_sex = LabelEncoder()
    # le_victim_age = LabelEncoder()
    le_victim_race = LabelEncoder()
    le_victim_eth = LabelEncoder()
    le_perp_sex = LabelEncoder()
    # le_perp_age = LabelEncoder()
    le_perp_race = LabelEncoder()
    le_perp_eth = LabelEncoder()
    le_relationship = LabelEncoder()
    le_weapon = LabelEncoder()

    # Apply label encoding
    df["Crime Solved"] = le_crime_solved.fit_transform(df["Crime Solved"])
    df["Victim Sex"] = le_victim_sex.fit_transform(df["Victim Sex"])
    # df['Victim Age'] = le_victim_age.fit_transform(df['Victim Age'])
    df["Victim Race"] = le_victim_race.fit_transform(df["Victim Race"])
    df["Victim Ethnicity"] = le_victim_eth.fit_transform(df["Victim Ethnicity"])
    df["Perpetrator Sex"] = le_perp_sex.fit_transform(df["Perpetrator Sex"])
    # df['Perpetrator Age'] = le_perp_age.fit_transform(df['Perpetrator Age'])
    df["Perpetrator Race"] = le_perp_race.fit_transform(df["Perpetrator Race"])
    df["Perpetrator Ethnicity"] = le_perp_eth.fit_transform(df["Perpetrator Ethnicity"])
    df["Relationship"] = le_relationship.fit_transform(df["Relationship"])
    df["Weapon"] = le_weapon.fit_transform(df["Weapon"])
    encoders_and_columns = {
        "Crime Solved": le_crime_solved,
        "Victim Sex": le_victim_sex,
        #    'Victim Age':le_victim_age,
        "Victim Race": le_victim_race,
        "Victim Ethnicity": le_victim_eth,
        "Perpetrator Sex": le_perp_sex,
        #    'Perpetrator Age':le_perp_age,
        "Perpetrator Race": le_perp_race,
        "Perpetrator Ethnicity": le_perp_eth,
        "Relationship": le_relationship,
        "Weapon": le_weapon,
        "columns": [
            "Crime Solved",
            "Victim Sex",
            "Victim Age",
            "Victim Race",
            "Victim Ethnicity",
            "Perpetrator Sex",
            "Perpetrator Age",
            "Perpetrator Race",
            "Perpetrator Ethnicity",
            "Relationship",
            "Weapon",
        ],
    }
    for column in df.columns:
        df[column] = df[column].astype(int)  # Try to convert the column to int
    return df, encoders_and_columns


@flow
def prep_flow(data_path: str, dest_path: str):
    unzip_file(data_path)
    # Load parquet files
    df = read_dataframe(os.path.join(data_path, "database.csv"))

    df, encoders_and_columns = preprocess(df)
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(df, os.path.join(dest_path, "data.pkl"))
    dump_pickle(encoders_and_columns, os.path.join(dest_path, "encoders.pkl"))
