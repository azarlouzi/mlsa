#ifndef _NESTED_SA_
#define _NESTED_SA_

#include "helpers.h"
#include "sa.h"

class Nested_Simulator {
public:
   void set_bias_parameter(double h);
   virtual double operator()() const=0;
protected:
   double h = 1;
   void verify_bias_parameter() const;
};

Risk_Measures nested_sa(IN     double            xi_0,
                        IN     double            c_0,
                        IN     double            alpha,
                        IN     double            h,
                        IN     long int          n,
                        IN     const Step&       step,
                        IN OUT Nested_Simulator& simulator);

double nested_sa_optimal_bias(double precision);
long int nested_sa_optimal_steps(double precision, const Step& step, double scaler);

#endif // _NESTED_SA_
