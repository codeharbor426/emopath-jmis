import pandas as pd


def load_dictionary(xlsx_file="data/dictionaries/Five Emotions Dictionaries.xlsx"):

    df = pd.read_excel(xlsx_file)

    dictionary = {
        '1_Anger': [str(i).strip() for i in df['1_Anger'] if not pd.isna(i)],
        '2_Frustration': [str(i).strip() for i in df['2_Frustration'] if not pd.isna(i)],
        '3_Disappointment': [str(i).strip() for i in df['3_Disappointment '] if not pd.isna(i)],
        '4_Helplessness': [str(i).strip() for i in df['4_Helplessness'] if not pd.isna(i)],
        '5_Anxiety': [str(i).strip() for i in df['5_Anxiety '] if not pd.isna(i)]
    }

    return dictionary