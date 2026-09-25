from utils.macro_indicators import get_macro_data

if __name__ == "__main__":
    df = get_macro_data()
    print(df.head())
