from typing import Callable, Dict, List, Tuple, Union
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

def plot_time_vs_rmse(title:str, algorithm_list:Tuple[str], accuracies:List[float],
                      d:Dict[str,Dict[str,Union[str,List[float]]]], outpath:str):
   labels = [r"$\frac{1}{%d}$" %int(math.ceil(1./accuracy)) for accuracy in accuracies]

   fig, ax = plt.subplots(figsize=FIGSIZE)
   #ax.invert_xaxis()
   ax.set_xscale("log")
   ax.set_yscale("log")
   #ax.grid(True, "both")

   for algorithm in algorithm_list:
      time_values, rmse_values = d[algorithm]["time"], d[algorithm]["rmse"]

      ax.plot(rmse_values, time_values, d[algorithm]["symbol"], label=algorithm, color=d[algorithm]["color"])
      alignment = "left" if algorithm == "Nested SA" else "right"
      xytext = (7,0) if algorithm == "Nested SA" else (-7,0)
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

def sub_process(filename:str, outpath:str, VaR:float, ES:float, ml_sa_focus:str)->None:
   def rmse(estimate_values:List[float], risk_measure:float)->float:
      return math.sqrt(mean(map(lambda x: pow(x - risk_measure, 2), estimate_values)))

   def average(time_values:List[float])->float:
      return mean(time_values)

   accuracies = []
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
         accuracy = float(d["accuracy"])
         if accuracy not in accuracies:
            accuracies.append(accuracy)

         time_sa_dict[accuracy].append(float(d["time_sa"]))
         time_nsa_dict[accuracy].append(float(d["time_nsa"]))
         time_mlsa_dict[accuracy].append(float(d["time_mlsa"]))
         estimate_VaR_sa_dict[accuracy].append(float(d["VaR_sa"]))
         estimate_VaR_nsa_dict[accuracy].append(float(d["VaR_nsa"]))
         estimate_VaR_mlsa_dict[accuracy].append(float(d["VaR_mlsa"]))
         estimate_ES_sa_dict[accuracy].append(float(d["ES_sa"]))
         estimate_ES_nsa_dict[accuracy].append(float(d["ES_nsa"]))
         estimate_ES_mlsa_dict[accuracy].append(float(d["ES_mlsa"]))

   average_time_sa = []
   average_time_nsa = []
   average_time_mlsa = []
   rmse_VaR_sa = []
   rmse_VaR_nsa = []
   rmse_VaR_mlsa = []
   rmse_ES_sa = []
   rmse_ES_nsa = []
   rmse_ES_mlsa = []

   accuracies = sorted(accuracies)[::-1]
   for accuracy in accuracies:
      average_time_sa.append(average(time_sa_dict[accuracy]))
      average_time_nsa.append(average(time_nsa_dict[accuracy]))
      average_time_mlsa.append(average(time_mlsa_dict[accuracy]))
      rmse_VaR_sa.append(rmse(estimate_VaR_sa_dict[accuracy], VaR))
      rmse_VaR_nsa.append(rmse(estimate_VaR_nsa_dict[accuracy], VaR))
      rmse_VaR_mlsa.append(rmse(estimate_VaR_mlsa_dict[accuracy], VaR))
      rmse_ES_sa.append(rmse(estimate_ES_sa_dict[accuracy], ES))
      rmse_ES_nsa.append(rmse(estimate_ES_nsa_dict[accuracy], ES))
      rmse_ES_mlsa.append(rmse(estimate_ES_mlsa_dict[accuracy], ES))

   if ml_sa_focus == ML_SA_VAR_FOCUS:
      plot_time_vs_rmse("Value-at-risk",
                        ("SA", "Nested SA", "Multilevel SA"),
                        accuracies,
                        {
                           "SA": {
                              "time": average_time_sa,
                              "rmse": rmse_VaR_sa,
                              "symbol": "-s",
                              "color": "tab:green",
                           }, "Nested SA": {
                              "time": average_time_nsa,
                              "rmse": rmse_VaR_nsa,
                              "symbol": "-^",
                              "color": "tab:blue",
                           }, "Multilevel SA": {
                              "time": average_time_mlsa,
                              "rmse": rmse_VaR_mlsa,
                              "symbol": "-o",
                              "color": "tab:orange",
                           },
                        },
                        outpath)
   else:
      plot_time_vs_rmse("Expected shortfall",
                        ("SA", "Nested SA", "Multilevel SA"),
                        accuracies,
                        {
                           "SA": {
                              "time"  : average_time_sa,
                              "rmse"  : rmse_ES_sa,
                              "symbol": "-s",
                              "color" : "tab:green",
                           }, "Nested SA": {
                              "time"  : average_time_nsa,
                              "rmse"  : rmse_ES_nsa,
                              "symbol": "-^",
                              "color" : "tab:blue",
                           }, "Multilevel SA": {
                              "time"  : average_time_mlsa,
                              "rmse"  : rmse_ES_mlsa,
                              "symbol": "-o",
                              "color" : "tab:orange",
                           },
                        },
                        outpath)

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
   print("VaR,ES")
   print(f"{VaR},{ES}")

   h_values = []
   centered_VaR_values = []
   centered_ES_values = []
   rescaled_VaR_values = []
   rescaled_ES_values = []
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
         h_values.append(h)
         centered_VaR = float(d["VaR"]) - VaR
         centered_ES = float(d["ES"]) - ES
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
   print("VaR,ES")
   print(f"{VaR},{ES}")

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
   print("VaR,ES")
   print(f"{VaR},{ES}")

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
