#include "nested_sa.h"
#include <cmath>
#include <cstdio>
#include <exception>

class Nested_Simulator_Exception: public std::exception {
public:
   Nested_Simulator_Exception(double h): h(h) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Simulation bias parameter should satisfy: h > 0. Got instead: h = %.10f", h);
      return msg;
   }
private:
   double h;
};

void Nested_Simulator::set_bias_parameter(double h) {
   this->h = h;
   verify_bias_parameter();
}

void Nested_Simulator::verify_bias_parameter() const {
   if (!(h > 0)) {
      throw Nested_Simulator_Exception(h);
   }
}

Risk_Measures nested_sa(IN     double            xi_0,
                        IN     double            c_0,
                        IN     double            alpha,
                        IN     double            h,
                        IN     long int          n,
                        IN     const Step&       step,
                        IN OUT Nested_Simulator& simulator) {
   simulator.set_bias_parameter(h);
   double xi = xi_0;
   double c = c_0;

   double X_h;
   for (long int i = 0L; i < n; i++) {
      X_h = simulator();
      c = c - H_2(alpha, c, xi, X_h)/double(i+1L);
      xi = xi - step(i+1L)*H_1(alpha, xi, X_h);
   }

   return Risk_Measures {
      .VaR = xi,
      .ES = c,
   };
}

double nested_sa_optimal_bias(double precision) {
   return precision;
}

long int nested_sa_optimal_steps(double precision, const Step& step, double scaler) {
   return sa_optimal_steps(precision, step, scaler);
}
