"""
This program is designed as a comprehensive recommendation system pipeline for Yelp data. It processes various datasets, extracts
features, and trains an XGBoost-based model to predict user-business ratings. Below is an overview of the methods and models
used, along with performance optimizations:

1. Data Processing:
   - The program uses PySpark RDDs to handle large datasets efficiently. Data is read from JSON and CSV files, parsed,
     and transformed using domain-specific classes (e.g., `BusinessData`, `UserData`).
   - Feature engineering is a key focus. For instance, attributes like `membership_years` and `num_categories` are derived
     to provide more meaningful inputs to the model.
   - The `create_dataset` function integrates features from multiple sources (users, businesses, reviews, tips, and photos)
     into a unified format for modeling.

2. Model Training:
   - The program employs XGBoost, a gradient boosting algorithm known for its performance and scalability in regression tasks.
   - To improve efficiency, hyperparameters are pre-optimized and stored in `ModelBasedConfig`. Features are normalized using
     `MinMaxScaler`, ensuring compatibility with the model's expectations.
   - By carefully selecting and dropping redundant features, the model focuses on the most impactful predictors, improving both
     training speed and prediction accuracy.

3. Error Analysis:
   - Predictions are evaluated using absolute error, categorized into bins for detailed distribution analysis.
   - This allows for identifying performance bottlenecks and understanding prediction quality across different ranges.

4. Creative Improvements:
   - Aggregating photo-related data (e.g., unique labels) and tip-related data (e.g., likes) adds a unique dimension to
     feature engineering.
   - Parallel processing using Spark RDDs ensures scalability for large datasets.
   - Modular design (with classes like `BusinessData`, `UserData`) enhances code readability and reuse.

"""

import csv
import json
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


class Path:
    yelp_train_processed: str = "yelp_train_processed.csv"
    yelp_val_processed: str = "yelp_val_processed.csv"


class DataReader:
    def __init__(self, sc: SparkContext):
        self.sc = sc

    def read_csv_spark(self, path: str):
        rdd = self.sc.textFile(path)
        header = rdd.take(1)[0]
        rdd = rdd.zipWithIndex().filter(lambda row_index: row_index[1] > 0).map(lambda row_index: row_index[0].split(","))
        return rdd, header.split(",")

    def read_json_spark(self, path: str):
        rdd = self.sc.textFile(path)
        return rdd.map(json.loads)



class BusinessData:
    name_related_keys = ["name", "neighborhood", "address"]
    attribute_related_keys = ["attributes", "categories", "hours"]
    location_related_keys = ["postal_code", "city", "state"]

    keys_to_delete = name_related_keys + attribute_related_keys + location_related_keys

    @staticmethod
    def parse_row(row: dict):
        row["num_attrs"] = 0 if not row.get("attributes") else len(row["attributes"])
        row["num_categories"] = (
            0 if not row.get("categories") else len(row["categories"].split(","))
        )
        row["stars"] = float(row.get("stars", 0))

        return row

    @staticmethod
    def generate_mapping(rdd):
        state_to_index = rdd.map(lambda r: r.get("state")).distinct().zipWithIndex().collectAsMap()
        city_to_index = rdd.map(lambda r: r.get("city")).distinct().zipWithIndex().collectAsMap()

        return state_to_index, city_to_index

    @staticmethod
    def process(rdd):
        transformed_rdd = (
            rdd.map(BusinessData.parse_row)
            .map(
                lambda row: (
                    row.get("business_id"),
                    (
                        row.get("stars"),
                        row.get("review_count"),
                        row.get("is_open"),
                        row.get("num_attrs"),
                        row.get("num_categories"),
                    ),
                )
            )
        )
        return transformed_rdd.cache().collectAsMap()


class UserData:
    user_specific_keys = ["name", "friends", "elite", "yelping_since"]
    compliment_specific_keys = [
        "compliment_hot",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_cool",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
    ]

    keys_to_delete = user_specific_keys + compliment_specific_keys
    compliment_keys = compliment_specific_keys

    lc = len(compliment_keys)

    @staticmethod
    def parse_row(row: dict):
        row["num_elite"] = 0 if row.get("elite") == "None" else len(row["elite"].split(","))
        row["num_friends"] = 0 if row.get("friends") == "None" else len(row["friends"].split(","))
        row["avg_compliment"] = sum(row.get(key, 0) for key in UserData.compliment_keys) / UserData.lc

        yelping_since_date = datetime.strptime(row.get("yelping_since"), "%Y-%m-%d")
        row["membership_years"] = (datetime.now() - yelping_since_date).days / 365.25

        row["average_stars"] = float(row.get("average_stars", 0))

        return row

    @staticmethod
    def process(rdd):
        transformed_rdd = (
            rdd.map(UserData.parse_row)
            .map(
                lambda row: (
                    row.get("user_id"),
                    (
                        row.get("review_count"),
                        row.get("useful"),
                        row.get("funny"),
                        row.get("cool"),
                        row.get("fans"),
                        row.get("average_stars"),
                        row.get("num_elite"),
                        row.get("num_friends"),
                        row.get("avg_compliment"),
                        row.get("membership_years"),
                    ),
                )
            )
        )
        return transformed_rdd.cache().collectAsMap()


class ReviewData:
    keys_to_delete = ["review_id", "date", "text"]

    @staticmethod
    def parse_row(row):
        filtered_row = {}
        for k, v in row.items():
            if k not in ReviewData.keys_to_delete:
                filtered_row[k] = v
        return filtered_row

    def process(rdd):
        def create_key_value(row):
            key = (row["user_id"], row["business_id"])
            value = (row["stars"], row["useful"], row["funny"], row["cool"], 1)
            return key, value

        def sum_values(x, y):
            return (
                x[0] + y[0],
                x[1] + y[1],
                x[2] + y[2],
                x[3] + y[3],
                x[4] + y[4],
            )

        def calculate_averages(values):
            return (
                values[0] / values[4],
                values[1] / values[4],
                values[2] / values[4],
                values[3] / values[4],
            )

        transformed_rdd = (
            rdd
            .map(create_key_value)
            .reduceByKey(sum_values)
            .mapValues(calculate_averages)
        )
        return transformed_rdd.cache().collectAsMap()



class TipData:
    keys_to_delete = ["date", "text"]

    @staticmethod
    def parse_row(row):
        filtered_row = {}
        for key, value in row.items():
            if key not in TipData.keys_to_delete:
                filtered_row[key] = value
        return filtered_row

    def process(rdd):
        def create_key_value_pair(row):
            return (row["user_id"], row["business_id"]), (row["likes"], 1)

        def aggregate_values(x, y):
            return x[0] + y[0], x[1] + y[1]

        transformed_rdd = (
            rdd
            .map(create_key_value_pair)
            .reduceByKey(aggregate_values)
        )
        return transformed_rdd.cache().collectAsMap()
class PhotoData:
    keys_to_delete = ["photo_id", "caption"]
    possible_labels = ["drink", "food", "inside", "menu", "outside"]

    @staticmethod
    def parse_row(row):
        return {k: v for k, v in row.items() if k not in set(PhotoData.keys_to_delete)}

    @staticmethod
    def process(rdd):
        def combine_values(acc, new_value):
            acc[0].append(new_value[0][0])
            return (acc[0], acc[1] + new_value[1])

        def transform_values(values):
            unique_labels = set(values[0])
            return (len(unique_labels), values[1])

        processed_rdd = (
            rdd
            .map(lambda row: (row["business_id"], ([row["label"]], 1)))
            .aggregateByKey(
                ([], 0),
                combine_values,
                lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
            )
            .mapValues(transform_values)
        )

        return processed_rdd.cache().collectAsMap()


class ModelBasedConfig:
    # Define components of the drop columns separately and combine them
    base_cols = ["user_id", "business_id", "rating"]
    review_cols = ["review_avg_stars", "useful", "funny", "cool"]
    # Additional commented-out attributes
    # attr_cols = ["num_attrs", "num_categories"]
    # engagement_cols = ["likes", "upvotes", "num_cat", "num_img"]
    drop_cols: list = base_cols + review_cols

    # Generate params using a dictionary comprehension
    params: dict = {
        key: value
        for key, value in zip(
            [
                "lambda",
                "alpha",
                "colsample_bytree",
                "subsample",
                "learning_rate",
                "max_depth",
                "random_state",
                "min_child_weight",
                "n_estimators",
            ],
            [
                9.92724463758443,
                0.2765119705933928,
                0.5,
                0.8,
                0.02,
                17,
                2020,
                101,
                250,
            ],
        )
    }

    pred_cols: list = ["user_id", "business_id", "prediction"]



def create_dataset(row, usr_dict, bus_dict, review_dict, tip_dict, img_dict):
    # Unpack row values with default for rating
    usr, bus = row if len(row) == 2 else (row[0], row[1])
    rating = row[2] if len(row) == 3 else None

    # Get values from review_dict with defaults
    review_data = review_dict.get((usr, bus), (None, 0, 0, 0))
    r_avg_stars, useful, funny, cool = review_data

    # Get values from usr_dict with defaults
    user_data = usr_dict.get(usr, (0, None, None, None, 0, 3.5, 0, 0, 0, None))
    (
        usr_review_count,
        usr_useful,
        usr_funny,
        usr_cool,
        usr_fans,
        usr_avg_stars,
        num_elite,
        num_friends,
        usr_avg_comp,
        membership_years,
    ) = user_data

    # Get values from bus_dict with defaults
    business_data = bus_dict.get(bus, (3.5, 0, None, None, None))
    bus_data_keys = ["bus_avg_stars", "bus_review_count", "bus_is_open", "num_attrs", "num_categories"]
    bus_avg_stars, bus_review_count, bus_is_open, num_attrs, num_categories = business_data

    # Get values from tip_dict with defaults
    tip_data_keys = ["likes", "upvotes"]
    likes, upvotes = tip_dict.get((usr, bus), (0, 0))

    # Get values from img_dict with defaults
    img_data_keys = ["num_cat", "num_img"]
    num_cat, num_img = img_dict.get(bus, (0, 0))

    # Combine all keys into a list for return
    user_keys = [usr, bus, r_avg_stars, useful, funny, cool, usr_review_count, usr_useful, usr_funny, usr_cool, usr_fans, usr_avg_stars, num_elite, num_friends, usr_avg_comp, membership_years]
    business_keys = [bus_avg_stars, bus_review_count, bus_is_open, num_attrs, num_categories]
    tip_keys = [likes, upvotes]
    img_keys = [num_cat, num_img]

    return tuple(user_keys + business_keys + tip_keys + img_keys + [float(rating) if rating is not None else None])



def save_data(data: list, output_file_name: str):
    header = ["user_id", "business_id", "prediction"]
    
    # Open the file in write mode
    f = open(output_file_name, "w", newline="")
    try:
        # Write the header
        f.write(",".join(header) + "\n")
        
        # Write each row of data
        for row in data:
            f.write(",".join(map(str, row)) + "\n")
    finally:
        f.close()
def process_data(folder_path: str, test_file_name: str):
    start_time = time.time()

    # Initialize Spark
    conf = SparkConf().setAppName("Competition: Recommendation system")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        data_reader = DataReader(spark)

        # Function to process individual datasets
        def process_dataset(file_name, processor=None):
            rdd = data_reader.read_json_spark(os.path.join(folder_path, file_name))
            return processor(rdd) if processor else rdd

        # Load and process all datasets
        usr_rdd = process_dataset("user.json", UserData.process)
        bus_rdd = process_dataset("business.json", BusinessData.process)
        review_rdd = process_dataset("review_train.json", ReviewData.process)
        tip_rdd = process_dataset("tip.json", TipData.process)
        img_rdd = process_dataset("photo.json", PhotoData.process)

        # Read train and validation datasets
        train_rdd, _ = data_reader.read_csv_spark(os.path.join(folder_path, "yelp_train.csv"))
        val_rdd, _ = data_reader.read_csv_spark(test_file_name)

        # Helper function to create processed datasets
        def process_rdd(input_rdd):
            return input_rdd.map(lambda row: create_dataset(row, usr_rdd, bus_rdd, review_rdd, tip_rdd, img_rdd))

        train_processed = process_rdd(train_rdd)
        val_processed = process_rdd(val_rdd)

        # Convert processed RDDs to Pandas DataFrame and save to CSV
        def save_to_csv(processed_rdd, output_path):
            column_names = [
                "user_id", "business_id", "review_avg_stars", "useful", "funny", "cool",
                "usr_review_count", "usr_useful", "usr_funny", "usr_cool", "usr_fans",
                "usr_avg_stars", "num_elite", "num_friends", "usr_avg_comp",
                "membership_years", "bus_avg_stars", "bus_review_count", "bus_is_open",
                "num_attrs", "num_categories", "likes", "upvotes", "num_cat",
                "num_img", "rating",
            ]
            df = pd.DataFrame(processed_rdd.collect(), columns=column_names)
            df.to_csv(output_path, index=False)

        save_to_csv(train_processed, Path.yelp_train_processed)
        save_to_csv(val_processed, Path.yelp_val_processed)

    except Exception as e:
        print(f"Exception occurred:\n{e}")

    finally:
        spark.stop()

    execution_time = time.time() - start_time
    print(f"Data Processing Duration: {execution_time} s\n")




def train_model(train_data_path: str, test_data_path: str):
    start_time = time.time()

    def load_and_prepare_data(file_path, drop_cols, scaler=None, is_training=False):
        df = pd.read_csv(file_path)
        X = df.drop(columns=drop_cols)
        if scaler:
            X = scaler.fit_transform(X) if is_training else scaler.transform(X)
        y = df["rating"] if "rating" in df.columns else None
        return X, y, df

    # Initialize min-max scaler
    scaler = MinMaxScaler()

    # Load and preprocess training data
    X_train_norm, y_train, _ = load_and_prepare_data(
        train_data_path, ModelBasedConfig.drop_cols, scaler, is_training=True
    )

    # Load and preprocess test data
    X_test_norm, _, val_df_processed = load_and_prepare_data(
        test_data_path, ModelBasedConfig.drop_cols, scaler
    )

    # Train and Predict using XGBoost Model
    def train_and_predict(X_train, y_train, X_test):
        model = XGBRegressor(**ModelBasedConfig.params)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    # Get predictions
    val_df_processed["prediction"] = train_and_predict(X_train_norm, y_train, X_test_norm)

    # Prepare output dataframe
    def filter_columns(df, columns):
        return df.loc[:, columns]

    pred_df = filter_columns(val_df_processed, ModelBasedConfig.pred_cols)

    execution_time = time.time() - start_time
    print(f"Model Training Time: {execution_time} s\n")

    return pred_df.values.tolist()



def get_error_distribution(test_data_path: str, output_file_name: str):
    def read_and_calculate_error(test_data_path, output_file_name):
        def read_file(file_path, columns=None):
            return pd.read_csv(file_path, usecols=columns) if columns else pd.read_csv(file_path)

        val_df = read_file(test_data_path, ["user_id", "business_id", "rating"])
        pred_df = read_file(output_file_name)
        pred_df = pred_df.assign(error=(val_df["rating"] - pred_df["prediction"]).abs())
        return pred_df

    def categorize_error(pred_df):
        def create_error_bins(errors, bins, labels):
            return pd.cut(errors, bins=bins, labels=labels, right=False)

        bins = [-np.inf, 1, 2, 3, 4, np.inf]
        labels = [">=0 and <1:", ">=1 and <2:", ">=2 and <3:", ">=3 and <4:", ">=4:"]
        pred_df["Error Distribution:"] = create_error_bins(pred_df["error"], bins, labels)
        return pred_df

    def compute_error_distribution(pred_df):
        return pred_df["Error Distribution:"].value_counts().sort_index()

    pred_df = read_and_calculate_error(test_data_path, output_file_name)
    pred_df = categorize_error(pred_df)
    error_distribution = compute_error_distribution(pred_df)

    print(error_distribution)


def main(folder_path: str, test_file_name: str, output_file_name: str):
    def execute_pipeline():
        # Process YELP Reviews Dataset
        process_data(folder_path, test_file_name)

        # Train Model Based Recommendation System
        return train_model(Path.yelp_train_processed, Path.yelp_val_processed)

    def save_and_evaluate(pred_data):
        # Save the predictions
        save_data(pred_data, output_file_name)

        # Get the error distribution
        get_error_distribution(Path.yelp_val_processed, output_file_name)

    start_time = time.time()
    pred_data = execute_pipeline()
    save_and_evaluate(pred_data)
    
    execution_time = time.time() - start_time
    print(f"Total Execution Time: {execution_time} s\n")

if __name__ == "__main__":
    def parse_arguments():
        if len(sys.argv) != 4:
            print("Usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>")
            sys.exit(1)
        return sys.argv[1], sys.argv[2], sys.argv[3]

    # Read input parameters
    folder_path, test_file_name, output_file_name = parse_arguments()

    main(folder_path, test_file_name, output_file_name)
