#include "swap_model.h"
#include "helpers.h"
#include <cmath>
#include <cstdio>

double Discount::operator()(double t) const {
   return std::exp(-r*t);
}

double Reset::operator()(int i) const {
   return Delta*i;
}

Swap_Simulator::Swap_Simulator(double r,
                               double S_0,
                               double kappa,
                               double sigma,
                               Time   Delta,
                               Time   T,
                               Time   delta,
                               double leg_0):
                               Simulator(),
                               r(r),
                               S_0(S_0),
                               kappa(kappa),
                               sigma(sigma),
                               Delta(time_to_years(Delta)),
                               T(time_to_years(T)),
                               delta(time_to_years(delta)),
                               leg_0(leg_0),
                               n(int(this->T/this->Delta)),
                               discount(r),
                               reset(this->Delta) {
   double tmp = 0;

   for (int i = 1; i <= n; i++) {
      tmp += discount(reset(i))*(this->Delta)*std::exp(kappa*reset(i-1));
   }
   nominal = leg_0/(S_0*tmp);

   tmp -= discount(reset(1))*(this->Delta);
   factor = nominal*S_0*tmp;
}

double Swap_Simulator::operator()() const {
   double simulation = factor*(doleans_dade(0, sigma*std::sqrt(delta)) - 1);
   return simulation;
}

Swap_Nested_Payoff::Swap_Nested_Payoff(double r,
                                       double S_0,
                                       double kappa,
                                       double Delta,
                                       double leg_0,
                                       int    n):
                                       r(r),
                                       S_0(S_0),
                                       kappa(kappa),
                                       Delta(Delta),
                                       leg_0(leg_0),
                                       n(n),
                                       discount(r),
                                       reset(Delta) {
   double tmp = 0;
   for (int i = 1; i <= n; i++) {
      tmp += discount(reset(i))*Delta*std::exp(kappa*reset(i-1));
   }
   nominal = leg_0/(S_0*tmp);
}

double Swap_Nested_Payoff::operator()(double y, double* z) const {
   double payoff = 0;
   double tmp = y;
   for (int i = 2; i <= n; i++) {
      tmp *= z[i-2];
      payoff += discount(reset(i))*Delta*std::exp(kappa*reset(i-1))*(tmp - 1);
   }
   payoff *= nominal*S_0;
   return payoff;
}

Swap_Nested_Simulator::Swap_Nested_Simulator(double r,
                                             double S_0,
                                             double kappa,
                                             double sigma,
                                             Time   Delta,
                                             Time   T,
                                             Time   delta,
                                             double leg_0):
                                             Nested_Simulator(),
                                             sigma(sigma),
                                             Delta(time_to_years(Delta)),
                                             T(time_to_years(T)),
                                             delta(time_to_years(delta)),
                                             n(int(this->T/this->Delta)),
                                             phi(r, S_0, kappa, this->Delta, leg_0, n) {}

double Swap_Nested_Simulator::operator()() const {
   double Y = doleans_dade(0, sigma*std::sqrt(delta));
   double Z[n-1];

   long int K = (long int) std::ceil(1./h);
   double X_h = 0;
   for (long int k = 0L; k < K; k++) {
      Z[0] = doleans_dade(0, sigma*std::sqrt(Delta-delta));
      for (int i = 1; i < n-1; i++) {
         Z[i] = doleans_dade(0, sigma*std::sqrt(Delta));
      }
      X_h += phi(Y, Z)/double(K);
   }

   return X_h;
}

Swap_ML_Simulator::Swap_ML_Simulator(double r,
                                     double S_0,
                                     double kappa,
                                     double sigma,
                                     Time   Delta,
                                     Time   T,
                                     Time   delta,
                                     double leg_0):
                                     ML_Simulator(),
                                     sigma(sigma),
                                     Delta(time_to_years(Delta)),
                                     T(time_to_years(T)),
                                     delta(time_to_years(delta)),
                                     n(int(this->T/this->Delta)),
                                     phi(r, S_0, kappa, this->Delta, leg_0, n) {}

ML_Simulations Swap_ML_Simulator::operator()() const {
   double Y = doleans_dade(0, sigma*std::sqrt(delta));
   double Z[n-1];

   long int K_coarse = (long int) std::ceil(1./h_coarse);
   double X_h_coarse = 0;
   for (long int k = 0L; k < K_coarse; k++) {
      Z[0] = doleans_dade(0, sigma*std::sqrt(Delta-delta));
      for (int i = 1; i < n-1; i++) {
         Z[i] = doleans_dade(0, sigma*std::sqrt(Delta));
      }
      X_h_coarse += phi(Y, Z)/double(K_coarse);
   }

   long int K_fine = (long int) std::ceil(1./h_fine);
   double X_h_fine = X_h_coarse*double(K_coarse)/double(K_fine);
   for (long int k = 0L; k < (K_fine - K_coarse); k++) {
      Z[0] = doleans_dade(0, sigma*std::sqrt(Delta-delta));
      for (int i = 1; i < n-1; i++) {
         Z[i] = doleans_dade(0, sigma*std::sqrt(Delta));
      }
      X_h_fine += phi(Y, Z)/double(K_fine);
   }

   return ML_Simulations {
      .coarse = X_h_coarse,
      .fine = X_h_fine,
   };
}
