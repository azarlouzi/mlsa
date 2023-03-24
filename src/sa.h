#ifndef _SA_
#define _SA_

#include "helpers.h"

struct Risk_Measures {
   double VaR = 0;
   double ES = 0;
};

class Simulator {
public:
   virtual double operator()() const=0;
};

class Step {
public:
   virtual double operator()(long int p) const=0;
   virtual long int inverse(double s) const=0;
};

class Gamma: public Step {
public:
   Gamma(double gamma_0 = 1, double beta = 1, long int smoothing = 0L):
      gamma_0(gamma_0), beta(beta), smoothing(smoothing) {}
   double operator()(long int p) const override;
   long int inverse(double s) const override;
private:
   double gamma_0, beta;
   long int smoothing;
};

double H_1(double alpha, double xi, double x);
double H_2(double alpha, double c, double xi, double x);

Risk_Measures sa(IN double           xi_0,
                 IN double           c_0,
                 IN double           alpha,
                 IN long int         n,
                 IN const Step&      step,
                 IN const Simulator& simulator);

long int sa_optimal_steps(double precision, const Step& step, double scaler);

#endif // _SA_
