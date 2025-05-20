functions {
  // Spectral density functions
  vector spd_se(vector omega, real alpha, real rho) {
    return alpha^2 * sqrt(2 * pi()) * rho * exp(-(rho^2 * omega^2)/2);
  }

  vector spd_matern32(vector omega, real alpha, real rho) {
    return alpha^2 * (12 * sqrt(3) / rho^3) * (3 / rho^2 + omega^2)^(-2);
  }

  vector spd_matern52(vector omega, real alpha, real rho) {
    return alpha^2 * (16 * 5^(5.0/2) / (3 * rho^5)) * (5 / rho^2 + omega^2)^(-3);
  }

  // Eigenvalues and eigenvectors
  vector eigenvalues(int M, real L) {
    vector[M] lambda;
    for (m in 1:M){
      lambda[m] = ((m * pi()) / (2 * L) )^2;
    }
    return lambda;
  }

  matrix eigenvectors(vector x, int M, real L, vector lambda) {
    int N = size(x);
    matrix[N,M] PHI;
    for (m in 1:M){
      for (n in 1:N){
        PHI[n,m] = sqrt(1/L) * sin(sqrt(lambda[m]) * (x[n] + L));
      }
    }
    return PHI;
  }

  // HSGP functions
  vector hsgp_se(vector x, real alpha, real rho, vector lambdas,
                  matrix PHI, vector z) {
    int N = rows(x);
    int M = cols(PHI);
    vector[N] f;
    matrix[M, M] Delta;
    // Spectral densities evaluated at the square root of the eigenvalues
    vector[M] spd = spd_se(sqrt(lambdas), alpha, rho);
    // Construct the diagonal matrix Delta
    Delta = diag_matrix(spd);
    // Compute the HSGP sample
    f = PHI * (Delta * z);
    return f;
  }

  // Add hsgp_matern32 and hsgp_matern52 if you intend to use them
  // For this example, we'll stick to hsgp_se as used in the transformed parameters block
}

data {
  int<lower=1> N;       // Number of observations
  vector[N] x;          // Standardized time points
  vector[N] y;          // PM2.5 observations
  real<lower=0> C;      // Boundary condition multiplier
  int<lower=1> M;       // Number of basis functions
}

transformed data {
  // Boundary condition
  real<lower=0> L = C * max(fabs(x)); // Use fabs for vector compatibility if x can be negative, max(abs(x)) from code
  
  // Compute the eigenvalues
  vector[M] lambdas = eigenvalues(M, L);
  
  // Compute the eigenvectors
  matrix[N, M] PHI = eigenvectors(x, M, L, lambdas);
}

parameters {
  real<lower=0> alpha;   // GP marginal standard deviation
  real<lower=0> rho;     // GP lengthscale
  real<lower=0> sigma;   // Observation noise standard deviation
  real beta_0;           // Intercept
  vector[M] z;           // Basis function coefficients (standard normal prior)
}

transformed parameters {
  vector[N] f = hsgp_se(x, alpha, rho, lambdas, PHI, z);
  vector[N] mu = beta_0 + f;
}

model {
  // Priors
  alpha ~ cauchy(0, 1);
  rho ~ inv_gamma(5, 1);     // As specified
  sigma ~ cauchy(0, 1);      // As specified
  beta_0 ~ normal(0, 2);     // As specified
  z ~ normal(0, 1);          // Standard normal prior for basis coefficients
  
  // Likelihood
  y ~ lognormal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;         // Posterior predictive samples
  // For predicting on new data points (days without observations), 
  // you would need to pass new_x, N_new, and compute PHI_new here
  // and then f_new = hsgp_se(new_x, alpha, rho, lambdas, PHI_new, z);
  // mu_new = beta_0 + f_new; y_new_rep = lognormal_rng(mu_new, sigma);

  for (n in 1:N) {
    log_lik[n] = lognormal_lpdf(y[n] | mu[n], sigma);
    y_rep[n] = lognormal_rng(mu[n], sigma);
  }
}
