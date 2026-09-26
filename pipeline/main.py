from utils.load_data import load_all_data

if __name__ == "__main__":
    df_macro, df_contract = load_all_data(contract_params = {"tickers": ["AAPL", "NVDA"]})
    