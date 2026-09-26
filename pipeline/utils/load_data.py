from .macro_indicators import get_macro_data
from .company_data import get_company_data

def load_all_data(macro_params: dict = None, contract_params: dict = None):
    """
    Load data for macroeconomic and contract usage. 
    Takes 2 dictionaries with parameter mappings for individual data loading functions.
    ``contract_params`` must take at least one specified ticker as a paramters. For example, 
    {"tickers": "AAPL"}, or {"tickers": ["AAPL", "NVDA"]}. 
    Returns 2 objects, df_macro and df_contract.
    """
    if macro_params is not None: df_macro = get_macro_data(**macro_params)
    else: df_macro = get_macro_data()

    if contract_params is not None: df_contract = get_company_data(**contract_params)
    else: df_contract = get_company_data()

    return df_macro, df_contract