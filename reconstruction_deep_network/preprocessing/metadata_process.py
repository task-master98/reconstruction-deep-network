import os
import pandas as pd


module_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(module_dir)
parent_dir = os.path.join(package_dir, "data")

dataset_name = "nyu_data"
dataset_dir = os.path.join(parent_dir, dataset_name)

train_metadata = os.path.join(dataset_dir, "nyu2_train.csv")
test_metadata = os.path.join(dataset_dir, "nyu2_test.csv")

train_df = pd.read_csv(train_metadata, names = ['color_path', 'depth_path'], header = None)
test_df = pd.read_csv(test_metadata, names = ['color_path', 'depth_path'], header = None)

def transform_paths_to_current(df: pd.DataFrame):
    for cols in df.columns:
        df[cols] = df[cols].apply(lambda x: x[5:])
    
    return df

def extract_class_names(df: pd.DataFrame):
    for cols in df.columns:
        df[f"{cols}_category"] = df[cols].apply(lambda x: x.split("/")[1])
    
    if df["color_path_category"].tolist() != df["depth_path_category"].tolist():
        raise ValueError("Categories are not the same")
    
    df = df.drop("color_path_category", axis = 1)
    df.rename(columns = {"depth_path_category": "category"}, inplace=True)
    return df


if __name__ == "__main__":
    train_df = transform_paths_to_current(train_df)
    train_df_new = extract_class_names(train_df)
    print(train_df_new.head())

    test_df_new = transform_paths_to_current(test_df)
    print(test_df_new.head())

    train_df_new.to_csv(os.path.join(dataset_dir, "nyu2_train_cleaned.csv"), index=False)
    test_df_new.to_csv(os.path.join(dataset_dir, "nyu2_test_cleaned.csv"), index=False)

    

