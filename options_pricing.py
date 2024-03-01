# -*- coding: utf-8 -*-

import csv
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# constant values
interest_rate = 0.03
fx_spot_volatility = 0.1
equity_fx_corr = 0.5
eur_to_usd = 1.10

def read_market_data(file):
  """
  Read market data from a CSV file.
    
  Args:
  file (str): The path to the CSV file containing market data.
    
  Returns:
  list: A list of dictionaries where each dictionary represents a row of data.
  """
  with open(file, 'r') as file:
    reader = csv.DictReader(file)
    return list(reader)

def read_trade_data(file):
  """
  Read trade data from a CSV file.
    
  Args:
  file (str): The path to the CSV file containing trade data.
    
  Returns:
  list: A list of dictionaries where each dictionary represents a row of data.
  """
  with open(file, 'r') as file:
    reader = csv.DictReader(file)
    return list(reader)

def option_price(S, K, T, r, sigma, option_type, currency):
  """
  Calculate the price of European options using the Black-Scholes model.

  Args:
  S (float): Spot price of the underlying asset.
  K (float): Strike price of the option.
  T (float): Time to expiration (in years).
  r (float): Risk-free interest rate.
  sigma (float): Volatility of the underlying asset.
  option_type (str): Type of option - 'call' or 'put'.
  currency (str): Currency type of the option payoff - 'REGULAR' (USD) or 'ODD' (EUR).

  Returns:
  float: Price of the option.

  Raises:
  ValueError: If an invalid option type is specified.
  """
  '''
  S: spot price, K: strike price, T: time to expiry, r: interest rate,
  sigma: volatility, option_type: call or put, currency: REGULAR (USD) or ODD (EUR)
  '''
  # adjust volatility for options with payoff in EUR
  if currency.lower() == 'odd':

    # the correlation affects the drift
    r = r + equity_fx_corr * sigma * fx_spot_volatility

    # fx spot volatility and correlation affects overall volatility
    sigma = np.sqrt(sigma ** 2 + fx_spot_volatility ** 2 + 2 * equity_fx_corr * sigma * fx_spot_volatility)

  # calculate d1 and d2
  d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)

  # Black-Scholes calculations for calls and puts respectively
  if option_type.lower() == 'call':
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
  elif option_type.lower() == 'put':
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  else:
    raise ValueError("Invalid option type. Please specify 'call' or 'put'.")

def calculate_pv_delta_vega(market_data, trade_data):
  """
  Calculate present value, equity delta, and equity vega for each trade.

  Args:
  market_data (list): A list of dictionaries representing market data, each containing 'underlying', 
  'spot_price', and 'volatility' keys.
  
  trade_data (list): A list of dictionaries representing trade data, each containing 'trade_id', 
  'quantity', 'underlying', 'expiry', 'payment_time', 'strike', 'call_put', and 'type' keys.

  Returns:
  list: A list of dictionaries containing results for each trade, with keys 'Trade ID', 'PV', 'Equity Delta', and 'Equity Vega'.
  """
  results = []
  for trade in trade_data:

    # match each trade with its market data point
    underlying_data = next(item for item in market_data if item["underlying"] == trade["underlying"])

    # extract necesarry values
    S = float(underlying_data["spot_price"])
    sigma = float(underlying_data["volatility"])
    K = float(trade["strike"])
    T = float(trade["expiry"])
    Q = float(trade["quantity"])
    T_p = float(trade['payment_time'])
    option_type = trade["call_put"]
    currency = trade["type"]

    # calculate the option price
    price = option_price(S, K, T, interest_rate, sigma, option_type, currency)

    # when calculating the option price, assume no payment delay
    # account for the payment delay when calculating PV by discounting
    payment_time_disc = np.exp(- interest_rate * (T_p - T))
    present_value = price * Q * payment_time_disc

    # PV sensitivity to the 1% relative increase in spot price
    delta = (option_price(S * 1.01, K, T, interest_rate, sigma, option_type, currency) - price) * Q * payment_time_disc

    # PV sensitivity to the 1% increase in volatility
    vega = (option_price(S, K, T, interest_rate, sigma + 0.01, option_type, currency) - price) * Q * payment_time_disc

    # append metrics to the results list, after rounding
    results.append({
      "Trade ID": trade["trade_id"],
      "PV": present_value.round(5),
      "Equity Delta": delta.round(5),
      "Equity Vega": vega.round(5)
    })
  return results

def write_results(results, filename):
  """
  Write the calculated results to a CSV file.

  Args:
  results (list of dict): List of dictionaries containing trade metrics.
  filename (str): Name of the output CSV file.
  """
  with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Trade ID", "PV", "Equity Delta", "Equity Vega"])
    writer.writeheader()
    writer.writerows(results)

def main(market_file, trade_file, result_file):
  """
  Main function to execute the entire process.

  Args:
  market_file (str): Path to the market data CSV file.
  trade_file (str): Path to the trade data CSV file.
  result_file (str): Path to the output result CSV file.
  """
  market_data = read_market_data(market_file)
  trade_data = read_trade_data(trade_file)
  results = calculate_pv_delta_vega(market_data, trade_data)
  write_results(results, result_file)

# main function will be called if the script is executed directly
# market_data.csv and trade_data.csv are assumed to be in the same directory
if __name__ == "__main__":
  main("market_data.csv", "trade_data.csv", "result.csv")


def implied_volatility(S, K, T, r, price, option_type, currency):
  """
  Back out implied volatility from an option price using Brent's method.

  Args:
  S (float): Spot price of the underlying asset.
  K (float): Strike price of the option.
  T (float): Time to expiration (in years).
  r (float): Risk-free interest rate.
  price (float): Option price.
  option_type (str): Type of option - 'call' or 'put'.
  currency (str): Currency type of the option payoff - 'REGULAR' (USD) or 'ODD' (EUR).

  Returns:
  float or str: Implied volatility if found, there might be no roots.

  Raises:
  ValueError: If the implied volatility is not found within the specified range or other error occurs.
  """
  try:
    # calculate option prices at two different volatility levels
    sigma_low = 0.1
    sigma_high = 0.5
    price_low = option_price(S, K, T, r, sigma_low, option_type, currency)
    price_high = option_price(S, K, T, r, sigma_high, option_type, currency)

    # check if the difference in option prices is below a threshold
    price_diff = abs(price_high - price_low)
    sensitivity_threshold = 0.001  # adjust as needed
    if price_diff < sensitivity_threshold:
      return "Option price is not sensitive to changes in volatility"

    # use Brent's method to find implied volatility
    # 0.3 is the initial guess since it is the average volatility in the data
    # the upper bound is 1
    implied_vol = brentq(lambda sigma: option_price(S, K, T, r, sigma, option_type, currency) - price, 0.3, 1)
    return implied_vol
  except ValueError:
      return "Implied volatility not found within the specified range or other error"

# examples:

# trade id 1:
# opt1 = option_price(71.16, 40, 3.00104, 0.03, 0.4639, 'call', 'regular')
# print(implied_volatility(71.16, 40, 3.00104, 0.03, opt1, 'call', 'regular'))

# trade id 4:
# opt2 = option_price(96.61, 102, 4.62112, 0.03, 0.3534, 'put', 'regular')
# print(implied_volatility(96.61, 102, 4.62112, 0.03, opt2, 'put', 'regular'))

# trade id 8:
# opt3 = option_price(121.24, 20, 2.80181, 0.03, 0.3893, 'put', 'odd')
# print(implied_volatility(121.24, 20, 2.80181, 0.03, opt3, 'put', 'odd'))
