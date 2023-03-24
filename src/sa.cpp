#include "sa.h"
#include <cmath>

double Gamma::operator()(long int n) const {
   return gamma_0/(std::pow(double(smoothing + n), beta));
}

long int Gamma::inverse(double s) const {
   return (long int) std::ceil(std::pow(s, -1./beta));
}

double H_1(double alpha, double xi, double x) {
   return 1 - heaviside(x-xi)/(1-alpha);
}

double H_2(double alpha, double c, double xi, double x) {
   return c - (xi + positive_part(x-xi)/(1-alpha));
}

Risk_Measures sa(IN double           xi_0,
                 IN double           c_0,
                 IN double           alpha,
                 IN long int         n,
                 IN const Step&      step,
                 IN const Simulator& simulator) {
   double xi = xi_0;
   double c = c_0;
   double X_0;
   for (long int i = 0L; i < n; i++) {
      X_0 = simulator();
      c = c - H_2(alpha, c, xi, X_0)/double(i+1L);
      xi = xi - step(i+1L)*H_1(alpha, xi, X_0);
   }
   return Risk_Measures {
      .VaR = xi,
      .ES = c,
   };
}

long int sa_optimal_steps(double precision, const Step& step, double scaler) {
   return (long int) (scaler*double(step.inverse(precision*precision)));
}
