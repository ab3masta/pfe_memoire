data {
  // numbers of things
  int<lower=1> N;  // number of observations
  int<lower=1> I;  // items,  number of questions  
  int<lower=1> S;  // subjects,  number of users 
  // data
  int<lower=1,upper=I> item[N];
  int<lower=1,upper=S> subject[N];
  int<lower=0,upper=1> grade[N];
}
parameters {
  // parameters
  real ability[S];             //  alpha: ability of student
  real difficulty[I];          //  beta: difficulty of question
  real delta;                   // mean student ability
}
model {
  ability ~ std_normal();         
  difficulty ~ std_normal();   
  delta ~ normal(0.75,1);
  for(n in 1:N)
    grade[n] ~ bernoulli_logit(ability[subject[n]] - difficulty[item[n]] + delta);
}
generated quantities {
  int<lower=0,upper=1> y_pred[N];
  vector[N] log_lik;
  for(n in 1:N)
    y_pred[n] = bernoulli_logit_rgn(ability[subject[n]] - difficulty[item[n]] + delta);
    log_lik[n] = bernoulli_logit_lpmf( grade[n] | ability[subject[n]] - difficulty[item[n]] + delta);
}