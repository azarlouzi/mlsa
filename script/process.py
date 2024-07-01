from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import argparse
from collections import defaultdict
import csv
import math
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from statistics import mean

FIGSIZE = (6,5)

ML_SA_VAR_FOCUS = "VaR"
ML_SA_ES_FOCUS = "ES"

def plot_biased_risk_measures(h_values:List[float],
                              centered_VaR_values:List[float],
                              centered_ES_values:List[float],
                              outpath:str, rescaled:bool=False)->None:
   fig, ax = plt.subplots(figsize=FIGSIZE)
   #ax.invert_xaxis()
   #ax.grid(True)

   if rescaled:
      ax.set_ylim((-6,8))
      ylabel = "Rescaled centered risk measure"
      title = "Rescaled centered risk measures"
   else:
      ylabel = "Centered risk measure"
      title = "Centered risk measures"

   ax.plot(h_values, centered_VaR_values, "-^", label="Value-at-risk")
   ax.plot(h_values, centered_ES_values, "-o", label="Expected shortfall")
   ax.set_xlabel("Bias parameter $h$", fontsize=16)
   ax.set_ylabel(ylabel, fontsize=16)
   ax.tick_params(labelsize=14)
   ax.legend(ncol=2, fontsize=14, frameon=False)
   ax.set_title(title, fontsize=16)

   plt.tight_layout()
   fig.savefig(os.path.join(outpath, title.replace(" ", "_").lower()+".pdf"), bbox_inches="tight", pad_inches=0)

def plot_time_vs_rmse(title:str, algorithm_list:Tuple[str], accuracy_dict:Dict[int,float],
                      d:Dict[str,Dict[str,Union[str,List[float]]]], outpath:str):
   labels = [r"$\frac{1}{%d}$" %int(math.ceil(1./accuracy_dict[cohort])) for cohort in range(1, len(accuracy_dict)+1)]

   fig, ax = plt.subplots(figsize=FIGSIZE)
   #ax.invert_xaxis()
   ax.set_xscale("log")
   ax.set_yscale("log")
   #ax.grid(True, "both")

   for algorithm in algorithm_list:
      time_values, rmse_values, fit_time_values = d[algorithm]["time"], d[algorithm]["rmse"], d[algorithm]["time_fit"]

      ax.plot(rmse_values, time_values, d[algorithm]["symbol"], label=algorithm, color=d[algorithm]["color"])
      pre, suf = d[algorithm]["color"].split(':')
      fit_color = ':'.join((pre, "dark " + suf))
      ax.plot(rmse_values, fit_time_values, "-"+d[algorithm]["symbol"], color=fit_color)
      alignment = "right" if algorithm in ("Nested SA", "Multilevel SA") else "left"
      xytext = (-7,0) if algorithm in ("Nested SA", "Multilevel SA") else (7,0)
      for i in range(len(rmse_values)):
         ax.annotate(labels[i], xy=(rmse_values[i], time_values[i]),
                     xytext=xytext, textcoords="offset points",
                     horizontalalignment=alignment, fontsize=12)

   ax.set_title(title, fontsize=16)
   ax.set_xlabel("RMSE", fontsize=14)
   ax.set_ylabel("Average execution time ($s$)", fontsize=14)
   ax.tick_params(labelsize=10)
   ax.legend(ncol=1, fontsize=14, frameon=False)

   plt.tight_layout()
   fname = title.lower().replace("-","_").replace(" ", "_") + "_comparison.pdf"
   fig.savefig(os.path.join(outpath, fname), bbox_inches="tight", pad_inches=0)

def plot_time_vs_accuracy(title:str, algorithm_list:Tuple[str], accuracy_dict:Dict[int,float],
                          d:Dict[str,Dict[str,Union[str,List[float]]]], outpath:str):
   accuracy_values = [accuracy_dict[cohort] for cohort in range(1, len(accuracy_dict)+1)]
   labels = [r"$\frac{1}{%d}$" %int(math.ceil(1./accuracy_dict[cohort])) for cohort in range(1, len(accuracy_dict)+1)]

   fig, ax = plt.subplots(figsize=FIGSIZE)
   #ax.invert_xaxis()
   ax.set_xscale("log")
   ax.set_yscale("log")
   #ax.grid(True, "both")

   for algorithm in algorithm_list:
      time_values, fit_time_values = d[algorithm]["time"], d[algorithm]["accuracy_time_fit"]

      ax.plot(accuracy_values, time_values, d[algorithm]["symbol"], label=algorithm, color=d[algorithm]["color"])
      pre, suf = d[algorithm]["color"].split(':')
      fit_color = ':'.join((pre, "dark " + suf))
      ax.plot(accuracy_values, fit_time_values, "-"+d[algorithm]["symbol"], color=fit_color)
      plt.xticks(accuracy_values, labels)
      alignment = "right" if algorithm in ("Nested SA", "Multilevel SA") else "left"
      xytext = (-7,0) if algorithm in ("Nested SA", "Multilevel SA") else (7,0)

   ax.set_title(title, fontsize=16)
   ax.set_xlabel("Prescribed accuracy $\epsilon$", fontsize=14)
   ax.set_ylabel("Average execution time ($s$)", fontsize=14)
   ax.tick_params(labelsize=10)
   ax.legend(ncol=1, fontsize=14, frameon=False)

   plt.tight_layout()
   fname = title.lower().replace("-","_").replace(" ", "_") + "_accuracy_comparison.pdf"
   fig.savefig(os.path.join(outpath, fname), bbox_inches="tight", pad_inches=0)

def fit_y_vs_x(y_values:List[float], x_values:List[float])->Tuple[float,List[float]]:
   [slope, intercept] = np.polyfit(np.log(x_values), np.log(y_values), 1)
   return slope, [np.exp(intercept)*np.power(x, slope) for x in x_values]

def rmse(estimate_values:List[float], risk_measure:float)->float:
   return math.sqrt(mean(map(lambda x: math.pow(x - risk_measure, 2), estimate_values)))

def average(values:List[float])->float:
   return mean(values)

def sub_process(filename:str, outpath:str, VaR:float, ES:float, ml_sa_focus:str)->None:
   cohort = 0

   accuracy_dict = {}
   time_sa_dict = defaultdict(list)
   time_nsa_dict = defaultdict(list)
   time_mlsa_dict = defaultdict(list)
   estimate_VaR_sa_dict = defaultdict(list)
   estimate_VaR_nsa_dict = defaultdict(list)
   estimate_VaR_mlsa_dict = defaultdict(list)
   estimate_ES_sa_dict = defaultdict(list)
   estimate_ES_nsa_dict = defaultdict(list)
   estimate_ES_mlsa_dict = defaultdict(list)

   with open(filename, "r") as instream:
      reader = csv.reader(instream)
      header = next(reader)
      for row in reader:
         if len(row) != len(header):
            continue
         d = dict(zip(header, row))
         if d.get("status") != "success":
            continue
         if d.get("#") == "1":
            cohort += 1
            accuracy = float(d["accuracy"])
            accuracy_dict[cohort] = accuracy

         time_sa_dict[cohort].append(float(d["time_sa"]))
         time_nsa_dict[cohort].append(float(d["time_nsa"]))
         time_mlsa_dict[cohort].append(float(d["time_mlsa"]))
         estimate_VaR_sa_dict[cohort].append(float(d["VaR_sa"]))
         estimate_VaR_nsa_dict[cohort].append(float(d["VaR_nsa"]))
         estimate_VaR_mlsa_dict[cohort].append(float(d["VaR_mlsa"]))
         estimate_ES_sa_dict[cohort].append(float(d["ES_sa"]))
         estimate_ES_nsa_dict[cohort].append(float(d["ES_nsa"]))
         estimate_ES_mlsa_dict[cohort].append(float(d["ES_mlsa"]))

   average_time_sa = []
   average_time_nsa = []
   average_time_mlsa = []
   rmse_VaR_sa = []
   rmse_VaR_nsa = []
   rmse_VaR_mlsa = []
   rmse_ES_sa = []
   rmse_ES_nsa = []
   rmse_ES_mlsa = []

   #accuracies = sorted(accuracies)[::-1]
   n_cohort = cohort
   for cohort in range(1, n_cohort+1):
      average_time_sa.append(average(time_sa_dict[cohort]))
      average_time_nsa.append(average(time_nsa_dict[cohort]))
      average_time_mlsa.append(average(time_mlsa_dict[cohort]))
      rmse_VaR_sa.append(rmse(estimate_VaR_sa_dict[cohort], VaR))
      rmse_VaR_nsa.append(rmse(estimate_VaR_nsa_dict[cohort], VaR))
      rmse_VaR_mlsa.append(rmse(estimate_VaR_mlsa_dict[cohort], VaR))
      rmse_ES_sa.append(rmse(estimate_ES_sa_dict[cohort], ES))
      rmse_ES_nsa.append(rmse(estimate_ES_nsa_dict[cohort], ES))
      rmse_ES_mlsa.append(rmse(estimate_ES_mlsa_dict[cohort], ES))

   VaR_sa_slope, VaR_sa_fit = fit_y_vs_x(average_time_sa, rmse_VaR_sa)
   ES_sa_slope, ES_sa_fit = fit_y_vs_x(average_time_sa, rmse_ES_sa)
   VaR_nsa_slope, VaR_nsa_fit = fit_y_vs_x(average_time_nsa, rmse_VaR_nsa)
   ES_nsa_slope, ES_nsa_fit = fit_y_vs_x(average_time_nsa, rmse_ES_nsa)
   VaR_mlsa_slope, VaR_mlsa_fit = fit_y_vs_x(average_time_mlsa, rmse_VaR_mlsa)
   ES_mlsa_slope, ES_mlsa_fit = fit_y_vs_x(average_time_mlsa, rmse_ES_mlsa)

   print("Time vs RMSE")
   if ml_sa_focus == ML_SA_VAR_FOCUS:
      print("VaR SA complexity exponent: %f" %VaR_sa_slope)
      print("VaR NSA complexity exponent: %f" %VaR_nsa_slope)
      print("VaR MLSA complexity exponent: %f" %VaR_mlsa_slope)
   else:
      print("ES SA complexity exponent: %f" %ES_sa_slope)
      print("ES NSA complexity exponent: %f" %ES_nsa_slope)
      print("ES MLSA complexity exponent: %f" %ES_mlsa_slope)

   print("\n")

   accuracies = [accuracy_dict[cohort] for cohort in range(1, len(accuracy_dict)+1)]

   VaR_sa_slope, VaR_sa_accuracy_fit = fit_y_vs_x(average_time_sa, accuracies)
   ES_sa_slope, ES_sa_accuracy_fit = fit_y_vs_x(average_time_sa, accuracies)
   VaR_nsa_slope, VaR_nsa_accuracy_fit = fit_y_vs_x(average_time_nsa, accuracies)
   ES_nsa_slope, ES_nsa_accuracy_fit = fit_y_vs_x(average_time_nsa, accuracies)
   VaR_mlsa_slope, VaR_mlsa_accuracy_fit = fit_y_vs_x(average_time_mlsa, accuracies)
   ES_mlsa_slope, ES_mlsa_accuracy_fit = fit_y_vs_x(average_time_mlsa, accuracies)

   print("Time vs accuracy")
   if ml_sa_focus == ML_SA_VAR_FOCUS:
      print("VaR SA complexity exponent: %f" %VaR_sa_slope)
      print("VaR NSA complexity exponent: %f" %VaR_nsa_slope)
      print("VaR MLSA complexity exponent: %f" %VaR_mlsa_slope)
   else:
      print("ES SA complexity exponent: %f" %ES_sa_slope)
      print("ES NSA complexity exponent: %f" %ES_nsa_slope)
      print("ES MLSA complexity exponent: %f" %ES_mlsa_slope)

   print("\n")

   summary = {
      "VaR": {
         "SA": {
            "time": average_time_sa,
            "rmse": rmse_VaR_sa,
            "time_fit": VaR_sa_fit,
            "accuracy_time_fit": VaR_sa_accuracy_fit,
            "symbol": "-s",
            "color": "xkcd:green",
         }, "Nested SA": {
            "time": average_time_nsa,
            "rmse": rmse_VaR_nsa,
            "time_fit": VaR_nsa_fit,
            "accuracy_time_fit": VaR_nsa_accuracy_fit,
            "symbol": "-^",
            "color": "xkcd:blue",
         }, "Multilevel SA": {
            "time": average_time_mlsa,
            "rmse": rmse_VaR_mlsa,
            "time_fit": VaR_mlsa_fit,
            "accuracy_time_fit": VaR_mlsa_accuracy_fit,
            "symbol": "-o",
            "color": "xkcd:orange",
         },
      }, "ES": {
         "SA": {
            "time": average_time_sa,
            "rmse": rmse_ES_sa,
            "time_fit": ES_sa_fit,
            "accuracy_time_fit": ES_sa_accuracy_fit,
            "symbol": "-s",
            "color": "xkcd:green",
         }, "Nested SA": {
            "time": average_time_nsa,
            "rmse": rmse_ES_nsa,
            "time_fit": ES_nsa_fit,
            "accuracy_time_fit": ES_nsa_accuracy_fit,
            "symbol": "-^",
            "color": "xkcd:blue",
         }, "Multilevel SA": {
            "time": average_time_mlsa,
            "rmse": rmse_ES_mlsa,
            "time_fit": ES_mlsa_fit,
            "accuracy_time_fit": ES_mlsa_accuracy_fit,
            "symbol": "-o",
            "color": "xkcd:orange",
         },
      },
   }

   if ml_sa_focus == ML_SA_VAR_FOCUS:
      plot_time_vs_rmse("Value-at-risk", ("SA", "Nested SA", "Multilevel SA"), accuracy_dict, summary["VaR"], outpath)
      plot_time_vs_accuracy("Value-at-risk", ("SA", "Nested SA", "Multilevel SA"), accuracy_dict, summary["VaR"], outpath)
   else:
      plot_time_vs_rmse("Expected shortfall", ("SA", "Nested SA", "Multilevel SA"), accuracy_dict, summary["ES"], outpath)
      plot_time_vs_accuracy("Expected shortfall", ("SA", "Nested SA", "Multilevel SA"), accuracy_dict, summary["ES"], outpath)

def compute_option_model_risk_measures(tau:float, alpha:float)->Tuple[float,float]:
   """ Compute exact values of the risk measures """
   VaR = tau*(math.pow(norm.ppf((1 - alpha)/2), 2) - 1)
   x = math.sqrt(1 + VaR/tau)
   ES = tau*(2*(x*norm.pdf(x) + norm.cdf(-x))/(1 - alpha) - 1)
   return VaR, ES

def compute_rate_swap_model_risk_measures(
   r:float, S_0:float, kappa:float, sigma:float,
   Delta:float, T:float, delta:float, # in days
   leg_0:float, alpha:float)->Tuple[float,float]:
   """ Compute exact values of the risk measures """

   def discount(t):
      return math.exp(-r*t)
   def reset(i):
      return i*Delta

   Delta = Delta/360.0
   T = T/360.0
   delta = delta/360.0
   n = int(T/Delta)

   nominal = leg_0/(S_0*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(1,n+1)))

   VaR = nominal*S_0*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(2,n+1))* \
         (math.exp(norm.ppf(alpha)*sigma*math.sqrt(delta)-sigma**2*delta/2)-1)

   omega = S_0 + VaR/(nominal*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(2,n+1)))
   d = (math.log(omega/S_0)-sigma**2*delta/2)/(sigma*math.sqrt(delta))
   ES = nominal*S_0*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(2,n+1))* \
        (alpha-norm.cdf(d))/(1-alpha)

   return VaR, ES

def process_test_case_0(filename:str, outpath:str, ml_sa_focus:str=ML_SA_VAR_FOCUS)->None:
   alpha = 0.975 # 0.0 < alpha < 1.0
   tau = 0.5 # 0.0 < tau < 1.0

   VaR, ES = compute_option_model_risk_measures(tau, alpha)
   print(f"VaR: {VaR}")
   print(f"ES: {ES}")
   print("\n")

   h_values = []
   VaR_dict = defaultdict(list)
   ES_dict = defaultdict(list)
   with open(filename, "r") as f:
      reader = csv.reader(f)
      header = next(reader)
      for row in reader:
         if len(row) != len(header):
            continue
         d = dict(zip(header, row))
         if d.get("status") != "success":
            continue
         h = float(d["h"])
         if h not in h_values:
            h_values.append(h)
         VaR_dict[h].append(float(d["VaR"]))
         ES_dict[h].append(float(d["ES"]))

   centered_VaR_values = []
   centered_ES_values = []
   rescaled_VaR_values = []
   rescaled_ES_values = []

   for h in h_values:
      centered_VaR = average(VaR_dict[h]) - VaR
      centered_ES = average(ES_dict[h]) - ES
      centered_VaR_values.append(centered_VaR)
      centered_ES_values.append(centered_ES)
      rescaled_VaR_values.append(centered_VaR/h)
      rescaled_ES_values.append(centered_ES/h)

   plot_biased_risk_measures(h_values, centered_VaR_values, centered_ES_values, outpath, rescaled=False)
   plot_biased_risk_measures(h_values, rescaled_VaR_values, rescaled_ES_values, outpath, rescaled=True)

def process_test_case_1(filename:str, outpath:str, ml_sa_focus:str=ML_SA_VAR_FOCUS)->None:
   alpha = 0.975 # 0.0 < alpha < 1.0
   tau = 0.5 # 0.0 < tau < 1.0

   VaR, ES = compute_option_model_risk_measures(tau, alpha)
   print(f"VaR: {VaR}")
   print(f"ES: {ES}")
   print("\n")

   sub_process(filename, outpath, VaR, ES, ml_sa_focus)

def process_test_case_2(filename:str, outpath:str, ml_sa_focus:str=ML_SA_VAR_FOCUS)->None:
   r = 0.02
   S_0 = 100.0 # in basis points
   kappa = 0.12
   sigma = 0.2
   Delta = 90.0 # in days
   T = 360.0 # in days
   delta = 7.0 # in days
   leg_0 = 1e4 # in basis points
   alpha = 0.85

   VaR, ES = compute_rate_swap_model_risk_measures(r, S_0, kappa, sigma, Delta, T, delta, leg_0, alpha)
   print(f"VaR: {VaR}")
   print(f"ES: {ES}")
   print("\n")

   sub_process(filename, outpath, VaR, ES, ml_sa_focus)

def processor(test_case:str)->Callable[[str,str],None]:
   PROCESSOR_MAP = {
      "0": process_test_case_0,
      "1": process_test_case_1,
      "2": process_test_case_2,
   }
   _processor = PROCESSOR_MAP.get(test_case)
   if _processor is None:
      raise NotImplementedError(f"Test set {test_case} is not implemented")
   return _processor

if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Process and plot SA results")
   p.add_argument("input_csv", help="csv file name")
   p.add_argument("output_path", help="Output path for figures")
   p.add_argument("--test_case", help="Test set to process", choices=["0","1","2"])
   p.add_argument("--ml_sa_focus", help="Multilevel stochastic approximation focus",
                  choices=[ML_SA_VAR_FOCUS, ML_SA_ES_FOCUS], default=ML_SA_VAR_FOCUS)
   args = p.parse_args()

   if not os.path.exists(args.input_csv) or not os.path.isfile(args.input_csv):
      raise FileNotFoundError(f"file {args.input_csv} not found")
   if not os.path.exists(args.output_path) or not os.path.isdir(args.output_path):
      raise IOError(f"directory {args.output_path} not found")

   processor(args.test_case)(args.input_csv, args.output_path, args.ml_sa_focus)
