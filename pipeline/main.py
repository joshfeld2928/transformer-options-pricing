from utils.macro_indicators import get_macro_data
from utils.company_data import get_company_data

if __name__ == "__main__":
    macro_data = get_macro_data()
    print(macro_data.head(5))

    company_data = get_company_data(tickers = ["AAPL"])
    print("\n", company_data.head())
