import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class Data:
    pass

    def to_dataframe(self):
        pass

    def to_augmented_dataframe(self):
        pass

    def join_dataframe(self):
        pass

    def smote_balance(self):
        pass

    def split_data(self):
        pass

    def generate_augmented_data(self):
        # ngambil dataframe
        pass

    def show_random_image(self, number):
        # number: jumlah gambar
        pass


def to_dataframe(data_dir):
    img_class_list = os.listdir(data_dir)
    img_path_list = []
    category_list = []
    for img_class in img_class_list:
        for img in os.listdir(os.path.join(data_dir, img_class)):
            img_path = os.path.join(data_dir, img_class, img)
            category_list.append(img_class)
            img_path_list.append(img_path)

    df = pd.DataFrame({
        'file_path': img_path_list,
        'category': category_list
    })

    return df


def join_dataframe(df1, df2):
    return pd.concat([df1, df2], ignore_index=True, sort=False)

def split_data(df):
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=1)  # 0.2
    df_train, df_test = train_test_split(df_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2
    return df_train, df_val, df_test

def smote_balance(batch_data, batch_label, img_size):
    sm = SMOTE(random_state=42)
    data, labels = sm.fit_resample(batch_data.reshape(-1, img_size*img_size*3), batch_label)
    data = data.reshape(-1, img_size, img_size, 3)
    return data, labels

