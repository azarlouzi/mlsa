#include "helpers.h"
#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"
#include "option_model.h"
#include "swap_model.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

void run_test_case_0() {
   double alpha = 0.975; // 0.0 < alpha < 1
   double tau = 0.5; // 0 < tau < 1
   double gamma_0 = 0.1; // gamma_0 > 0
   double beta = 1.0; // 0.0 < beta <= 1
   long int smoothing = 10000L;

   double xi_0 = 2.0;
   double c_0 = 2.9;

   Gamma gamma(gamma_0, beta, smoothing);
   Option_Nested_Simulator nested_simulator(tau); 
   Risk_Measures estimates;

   double h_values[] = {
      1./10.,
      1./20.,
      1./50.,
      1./100.,
      1./200.,
   }; // h = 1/K, K >= 1
   long int n = 1000000L; // n >> 1

   int n_runs = 200;
   std::printf("#,h,status,VaR,ES\n");
   for (double h: h_values) {
      for (int i = 0; i < n_runs; i++) {
         try {
            estimates = nested_sa(xi_0, c_0, alpha, h, n, gamma, nested_simulator);
            std::printf("%d,%.10f,success,%f,%f\n", i+1, h, estimates.VaR, estimates.ES);
         } catch (const std::exception& e) {
            std::printf("%d,%.10f,failure,%s\n", i+1, h, e.what());
         }
      }
   }
}

struct ML_Setting {
   double h_0;
   double M;
   int L;
   double gamma_0;
   long int smoothing;
};

void run_test_case_1(ML_SA_Focus ml_sa_focus = ml_sa_var_focus) {
   double alpha = 0.975; // 0.0 < alpha < 1
   double tau = 0.5; // 0 < tau < 1

   Option_Simulator        simulator        (tau);
   Option_Nested_Simulator nested_simulator (tau);
   Option_ML_Simulator     ml_simulator     (tau);

   int max_L = 20;
   double h[max_L+1];
   long int N[max_L+1];
   Loss_Model model {.concentration = power_concentration, .p = 11,};

   ML_Setting ml_settings[5];
   if (ml_sa_focus == ml_sa_var_focus) {
      ml_settings[0] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 1, .gamma_0 = 2, .smoothing = 2500L,}; // h_L = 1/32
      ml_settings[1] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 1, .gamma_0 = 2, .smoothing = 4000L,}; // h_L = 1/64
      ml_settings[2] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 2, .gamma_0 = 0.75, .smoothing = 9000L,}; // h_L = 1/128
      ml_settings[3] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 3, .gamma_0 = 0.25, .smoothing = 10000L,}; // h_L = 1/256
      ml_settings[4] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 4, .gamma_0 = 0.09, .smoothing = 10000L,}; // h_L = 1/512
   } else {
      ml_settings[0] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 1, .gamma_0 = 0.1, .smoothing = 10000L,}; // h_L = 1/32
      ml_settings[1] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 1, .gamma_0 = 0.1, .smoothing = 10000L,}; // h_L = 1/64
      ml_settings[2] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 2, .gamma_0 = 0.1, .smoothing = 10000L,}; // h_L = 1/128
      ml_settings[3] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 3, .gamma_0 = 0.1, .smoothing = 20000L,}; // h_L = 1/256
      ml_settings[4] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 4, .gamma_0 = 0.1, .smoothing = 25000L,}; // h_L = 1/512
   };

   double beta = 1.0; // 0.0 < beta <= 1
   Gamma gamma_sa;
   Gamma gamma_nsa;
   Gamma gamma_mlsa;

   if (ml_sa_focus == ml_sa_var_focus) {
      gamma_sa = Gamma(1, beta, 100L);
      gamma_sa = Gamma(1, beta, 100L);
   } else {
      gamma_sa = Gamma(0.1, beta, 25000L);
      gamma_sa = Gamma(0.1, beta, 25000L);
   };

   double h_L;
   double scaler = ml_sa_focus == ml_sa_var_focus ? 1 : 100;
   long int n; // n >> 1

   double xi_0 = 2.0;
   double c_0 = 2.9;
   double time_sa, time_nsa, time_mlsa;
   Risk_Measures risk_measures_sa, risk_measures_nsa, risk_measures_mlsa;

   int n_runs = 200;
   std::printf("#,accuracy,status,time_sa,time_nsa,time_mlsa,VaR_sa,ES_sa,VaR_nsa,ES_nsa,VaR_mlsa,ES_mlsa\n");
   for (ML_Setting ml_setting: ml_settings) {
      for (int i = 0; i < n_runs; i++) {
         try {
            // ML SA
            gamma_mlsa = Gamma(ml_setting.gamma_0, beta, ml_setting.smoothing);
            configure_ml_sa(ml_sa_focus, beta, ml_setting.h_0, ml_setting.M, ml_setting.L, scaler, model, h,  N, h_L);
            tik();
            risk_measures_mlsa = ml_sa(xi_0, c_0, alpha, ml_setting.L, h, N, gamma_mlsa, nested_simulator, ml_simulator);
            time_mlsa = tok();

            // Nested SA
            n = nested_sa_optimal_steps(h_L, gamma_nsa, scaler);
            tik();
            risk_measures_nsa = nested_sa(xi_0, c_0, alpha, h_L,  n, gamma_nsa, nested_simulator);
            time_nsa = tok();

            // SA
            n = sa_optimal_steps(h_L, gamma_sa, scaler);
            tik();
            risk_measures_sa = sa(xi_0, c_0, alpha,  n, gamma_sa, simulator);
            time_sa = tok();

            std::printf("%d,%.15f,success,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        i+1, h_L, time_sa, time_nsa, time_mlsa,
                        risk_measures_sa.VaR, risk_measures_sa.ES,
                        risk_measures_nsa.VaR, risk_measures_nsa.ES,
                        risk_measures_mlsa.VaR, risk_measures_mlsa.ES);
         } catch (const std::exception& e) {
            std::printf("%d,%.15f,failure,%s\n", i+1, h_L, e.what());
         }
      }
   }
}

void run_test_case_2(ML_SA_Focus ml_sa_focus = ml_sa_var_focus) {
   double r = 0.02;
   double S_0 = 100.0; // in basis points
   double kappa = 0.12;
   double sigma = 0.2;
   Time Delta = Time {y: 0, m: 3, d: 0};
   Time T = Time {y: 1, m: 0, d: 0}; 
   Time delta = Time {y: 0, m: 0, d: 7};
   double leg_0 = 1e4; // in basis points
   double alpha = 0.85; // 0.0 < alpha < 1

   Swap_Simulator        simulator        (r, S_0, kappa, sigma, Delta, T, delta, leg_0);
   Swap_Nested_Simulator nested_simulator (r, S_0, kappa, sigma, Delta, T, delta, leg_0);
   Swap_ML_Simulator     ml_simulator     (r, S_0, kappa, sigma, Delta, T, delta, leg_0);

   int max_L = 20;
   double h[max_L+1];
   long int N[max_L+1];
   Loss_Model model {.concentration = power_concentration, .p = 8};

   ML_Setting ml_settings[5];
   if (ml_sa_focus == ml_sa_var_focus) {
      ml_settings[0] = ML_Setting {.h_0 = 1./8, .M = 2, .L = 2, .gamma_0 = 6, .smoothing = 10L,}; // h_L = 1/32
      ml_settings[1] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 2, .gamma_0 = 20, .smoothing = 500L,}; // h_L = 1/64
      ml_settings[2] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 3, .gamma_0 = 21, .smoothing = 1000L,}; // h_L = 1/128
      ml_settings[3] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 4, .gamma_0 = 20, .smoothing = 2000L}; // h_L = 1/256
      ml_settings[4] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 5, .gamma_0 = 21, .smoothing = 3000L}; // h_L = 1/512
   } else {
      ml_settings[0] = ML_Setting {.h_0 = 1./8, .M = 2, .L = 2, .gamma_0 = 5, .smoothing = 10L,}; // h_L = 1/32
      ml_settings[1] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 2, .gamma_0 = 20, .smoothing = 500L,}; // h_L = 1/64
      ml_settings[2] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 3, .gamma_0 = 20, .smoothing = 500L,}; // h_L = 1/128
      ml_settings[3] = ML_Setting {.h_0 = 1./16, .M = 2, .L = 4, .gamma_0 = 20, .smoothing = 750L,}; // h_L = 1/256
      ml_settings[4] = ML_Setting {.h_0 = 1./32, .M = 2, .L = 4, .gamma_0 = 50, .smoothing = 2000L,}; // h_L = 1/512
   };

   double beta = 1.0; // 0.0 < beta <= 1
   Gamma gamma_sa (100, beta, 0L); // SA
   Gamma gamma_nsa (50, beta, 0L); // Nested SA
   Gamma gamma_mlsa; // Multilevel SA

   double h_L;
   double scaler = ml_sa_focus == ml_sa_var_focus ? 1 : 20;
   long int n; // n >> 1

   double xi_0 = 200;
   double c_0 = 330;
   double time_sa, time_nsa, time_mlsa;
   Risk_Measures risk_measures_sa, risk_measures_nsa, risk_measures_mlsa;

   int n_runs = 200;
   std::printf("#,accuracy,status,time_sa,time_nsa,time_mlsa,VaR_sa,ES_sa,VaR_nsa,ES_nsa,VaR_mlsa,ES_mlsa\n");
   for (ML_Setting ml_setting: ml_settings) {
      for (int i = 0; i < n_runs; i++) {
         try {
            // ML SA
            gamma_mlsa = Gamma(ml_setting.gamma_0, beta, ml_setting.smoothing);
            configure_ml_sa(ml_sa_focus, beta, ml_setting.h_0, ml_setting.M, ml_setting.L, scaler, model, h, N, h_L);
            tik();
            risk_measures_mlsa = ml_sa(xi_0, c_0, alpha, ml_setting.L, h, N, gamma_mlsa, nested_simulator, ml_simulator);
            time_mlsa = tok();

            // Nested SA
            n = nested_sa_optimal_steps(h_L, gamma_nsa, scaler);
            tik();
            risk_measures_nsa = nested_sa(xi_0, c_0, alpha, h_L, n, gamma_nsa, nested_simulator);
            time_nsa = tok();

            // SA
            n = sa_optimal_steps(h_L, gamma_sa, scaler);
            tik();
            risk_measures_sa = sa(xi_0, c_0, alpha, n, gamma_sa, simulator);
            time_sa = tok();

            std::printf("%d,%.15f,success,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                        i+1, h_L, time_sa, time_nsa, time_mlsa,
                        risk_measures_sa.VaR, risk_measures_sa.ES,
                        risk_measures_nsa.VaR, risk_measures_nsa.ES,
                        risk_measures_mlsa.VaR, risk_measures_mlsa.ES);
         } catch (const std::exception& e) {
            std::printf("%d,%.15f,failure,%s\n", i+1, h_L, e.what());
         }
      }
   }
}

void display_help(const char* name) {
   std::printf("Usage: %s [-h|--help|[--test_case {0,1,2}][--ml_sa_focus {VaR,ES}]]\n", name);
   std::printf("Options:\n");
   std::printf("-h, --help             Display this usage documentation\n");
   std::printf("--test_case   {0,1,2}  Test set to run; default: 1\n");
   std::printf("--ml_sa_focus {VaR,ES} Multilevel stochastic approximation focus; default: VaR\n");
}

class Parameter_Exception: public std::exception {
public:
   Parameter_Exception(const char* parameter_name, const char* expected_behavior, const char* encountered_instead):
      parameter_name(parameter_name), expected_behavior(expected_behavior), encountered_instead(encountered_instead) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Expected %s to satisfy: %s. Encountered instead: %s.", parameter_name, expected_behavior, encountered_instead);
      return msg;
   }
private:
   const char* parameter_name;
   const char* expected_behavior;
   const char* encountered_instead;
};

enum Test_Case {
   test_case_0 = 0,
   test_case_1 = 1,
   test_case_2 = 2,
};

Test_Case convert_string_to_test_case(std::string test_case_string) {
   if (test_case_string == "0") {
      return test_case_0;
   } else if (test_case_string == "1") {
      return test_case_1;
   } else if (test_case_string == "2") {
      return test_case_2;
   } else {
      throw Parameter_Exception("test_case", "test_case in {0,1,2}", test_case_string.c_str());
   }
}

ML_SA_Focus convert_string_to_ml_sa_focus(std::string ml_sa_focus_string) {
   if (ml_sa_focus_string == "VaR") {
      return ml_sa_var_focus;
   } else if (ml_sa_focus_string == "ES") {
      return ml_sa_es_focus;
   } else {
      throw Parameter_Exception("ml_sa_focus", "ml_sa_focus in {VaR, ES}", ml_sa_focus_string.c_str());
   }
}

void run(Test_Case test_case, ML_SA_Focus ml_sa_focus) {
   switch (test_case) {
      case test_case_0:
         run_test_case_0();
         break;
      case test_case_1:
         run_test_case_1(ml_sa_focus);
         break;
      case test_case_2:
         run_test_case_2(ml_sa_focus);
         break;
   }
}

int main(int argc, char* argv[]) {
    if ((argc < 2) ||
        (argc == 2 && (std::string(argv[1]) == "-h" ||
                       std::string(argv[1]) == "--help"))) {
      display_help(argv[0]);
      return 1;
   }

   Test_Case test_case = test_case_1;
   ML_SA_Focus ml_sa_focus = ml_sa_var_focus;

   if (argc < 2) {
      run(test_case, ml_sa_focus);
      return 0;
   }

   std::string test_case_string = "1";
   std::string ml_sa_focus_string = "VaR";
   try {
      for (int i = 1; i < argc; ) {
         if (std::string(argv[i]) == "--test_case") {
            test_case_string = std::string(argv[i+1]);
            i = i + 2;
         } else if (std::string(argv[i]) == "--ml_sa_focus") {
            ml_sa_focus_string = std::string(argv[i+1]);
            i = i + 2;
         } else {
            i = i + 1;
         }
      }
      test_case = convert_string_to_test_case(test_case_string);
      ml_sa_focus = convert_string_to_ml_sa_focus(ml_sa_focus_string);
   } catch(const std::exception& e) {
      std::printf("%s\n\n", e.what());
      display_help(argv[0]);
      return 1;
   }

   run(test_case, ml_sa_focus);
   return 0;
}
