#include "option_model.h"
#include "helpers.h"
#include <cmath>

Option_Simulator::Option_Simulator(double tau):
   Simulator(), tau(tau) {}

double Option_Simulator::operator()() const {
   double Y = gaussian();
   return tau*(std::pow(Y, 2) - 1);
}

Option_Nested_Payoff::Option_Nested_Payoff(double tau):
   tau(tau) {}

double Option_Nested_Payoff::operator()(double y, double z) const {
   return -std::pow(std::sqrt(tau)*y + std::sqrt(1-tau)*z, 2);
}

Option_Nested_Simulator::Option_Nested_Simulator(double tau):
   Nested_Simulator(), phi(tau) {}

double Option_Nested_Simulator::operator()() const {
   long int K = (long int) (std::ceil(1./h));
   double X_h = -1;
   double Y = gaussian();
   for (long int k = 0L; k < K; k++) {
      X_h -= phi(Y, gaussian())/double(K);
   }
   return X_h;
}

Option_ML_Simulator::Option_ML_Simulator(double tau):
   ML_Simulator(), phi(tau) {}

ML_Simulations Option_ML_Simulator::operator()() const {
   double Y = gaussian();

   long int K_coarse = (long int) std::ceil(1./h_coarse);
   double X_h_coarse = -1;
   for (long int k = 0L; k < K_coarse; k++) {
      X_h_coarse -= phi(Y, gaussian())/double(K_coarse);
   }

   long int K_fine = (long int) std::ceil(1./h_fine);
   double X_h_fine = -1 + (X_h_coarse + 1)*double(K_coarse)/double(K_fine);
   for (long int k = 0L; k < (K_fine - K_coarse); k++) {
      X_h_fine -= phi(Y, gaussian())/double(K_fine);
   }

   return ML_Simulations {
      .coarse = X_h_coarse,
      .fine = X_h_fine,
   };
}
