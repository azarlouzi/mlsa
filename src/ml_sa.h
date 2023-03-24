#ifndef _ML_SA_
#define _ML_SA_

#include "helpers.h"
#include "nested_sa.h"
#include "sa.h"

struct ML_Simulations {
   double coarse;
   double fine;
};

class ML_Simulator {
public:
   void set_bias_parameters(double h_coarse, double h_fine);
   virtual ML_Simulations operator()() const=0;
protected:
   double h_coarse = 1, h_fine = 0.5;
   void verify_bias_parameters() const;
};

enum Loss_Concentration {
   power_concentration,
   gaussian_concentration,
   lipschitz_concentration,
};

struct Loss_Model {
   Loss_Concentration concentration;
   double p = 0; // p > 1, exponent for power concentration
};

enum ML_SA_Focus {
   ml_sa_var_focus = 1,
   ml_sa_es_focus = 2,
};

void configure_ml_sa(IN     ML_SA_Focus       ml_sa_focus,
                     IN     double            beta,
                     IN     double            h_0,
                     IN     double            M,
                     IN     int               L,
                     IN     double            scaler,
                     IN     const Loss_Model& loss_model,
                        OUT double*           h,
                        OUT long int*         N,
                        OUT double&           h_L);

Risk_Measures ml_sa(IN     double            xi_0,
                    IN     double            c_0,
                    IN     double            alpha,
                    IN     int               L,
                    IN     const double*     h,
                    IN     const long int*   N,
                    IN     const Step&       step,
                    IN OUT Nested_Simulator& simulator,
                    IN OUT ML_Simulator&     ml_simulator);

int ml_sa_optimal_layers(double precision, double h_0, double M);

#endif // _ML_SA_
