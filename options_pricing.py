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

# helper functions to read market and trade data from csv
def read_market_data(file):
  with open(file, 'r') as file:
    reader = csv.DictReader(file)
    return list(reader)

def read_trade_data(file):
  with open(file, 'r') as file:
    reader = csv.DictReader(file)
    return list(reader)

# Black-Scholes formula for calculating price of European options
def option_price(S, K, T, r, sigma, option_type, currency):
  '''
  S: spot price, K: strike price, T: time to expiry, r: interest rate,
  sigma: volatility, option_type: call or put, currency: REGULAR (USD) or ODD (EUR)
  '''
  # to handle the case when payoff is in EUR
  if currency.lower() == 'odd':

    # fx spot volatility and correlation affects overall volatility
    sigma = np.sqrt(sigma ** 2 + fx_spot_volatility ** 2 - 2 * equity_fx_corr * sigma * fx_spot_volatility)

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

# function to calculate present value, equity delta, and equity vega
def calculate_pv_delta_vega(market_data, trade_data):
  '''
  market_data: table containing underlying, spot_price and volatility columns
  trade_data: table containing trade_id, Q, underlying, T, T_p, K, call_put and type columns
  '''
  results = []
  for trade in trade_data:

    # match each trade with its market data point
    underlying_data = next(item for item in market_data if item["underlying"] == trade["underlying"])

    # extracting the necesarry values
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

    # when calculating the option price, we assume no payment delay
    # here, we account for the payment delay by discounting
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

# write results to CSV
def write_results(results, filename):
  with open(filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["Trade ID", "PV", "Equity Delta", "Equity Vega"])
    writer.writeheader()
    writer.writerows(results)

# main function to put everything together
def main(market_file, trade_file, result_file):
  market_data = read_market_data(market_file)
  trade_data = read_trade_data(trade_file)
  results = calculate_pv_delta_vega(market_data, trade_data)
  write_results(results, result_file)

# main function will be called if the script is executed directly
# market_data.csv and trade_data.csv are assumed to be in the same directory
if __name__ == "__main__":
  main("market_data.csv", "trade_data.csv", "result.csv")

# function to back out implied volatility
def implied_volatility(S, K, T, r, price, option_type, currency):
  '''
  S: spot price, K: strike price, T: time to expiry, r: interest rate,
  price: option_price, option_type: call or put, currency: REGULAR (USD) or ODD (EUR)
  '''
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
