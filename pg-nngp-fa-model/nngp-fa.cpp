#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>

#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <Rinternals.h>

using namespace std;

#include "../libs/kvpar.h"

#include <time.h>
#include <sys/time.h>
#define CPUTIME (SuiteSparse_time ( ))

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

extern "C" {
  extern void dgemm_(const char *transa, const char *transb,
		    const int *m, const int *n, const int *k,
		    const double *alpha, const double *a,
		    const int *lda, const double *b, const int *ldb,
		    const double *beta, double *c, const int *ldc);
   
  extern void  dcopy_(const int *n, const double *dx, const int *incx, double *dy, const int *incy);
  
  extern int dpotrf_(const char *uplo, int *n, double *a, int *lda, int *info);

  extern int dpotri_(const char *uplo, int *n, double *a, int *lda, int *info);

  extern void dsymm_(const char *side, const char *uplo, const int *m,
		     const int *n, const double *alpha,
		     const double *a, const int *lda,
		     const double *b, const int *ldb,
		     const double *beta, double *c, const int *ldc);

  extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);

  extern void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
		     const double *a, const int *lda, const double *x, const int *incx,
		     const double *beta, double *y, const int *incy);
  
  extern void dsymv_(const char *uplo, const int *n, const double *alpha, const double *a, const int *lda,
		    const double *x, const int *incx, const double *beta, double *y, const int *incy);

  extern void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);

  extern void dtrmv_(const char *uplo, const char *transa, const char *diag, const int *n,
		     const double *a, const int *lda, double *b, const int *incx);

}

//rpg stuff

// Mathematical constants computed using Wolfram Alpha
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

// Generate exponential distribution random variates
double exprnd(double mu){
  return -mu * (double)std::log(1.0 - (double)runif(0.0,1.0));
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t){
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }    
  return (double)exp(f);
}


// Generate inverse gaussian random variates
double randinvg(double mu){
  // sampling
  double u = rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - (double)std::sqrt(4.0*mu*V + mu*mu * V*V) );
  
  if(runif(0.0,1.0) > mu /(mu+out)) {    
    out = mu*mu / out; 
  }    
  return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma(){
  double c = MATH_PI_2;
  double X, gX;
  
  bool done = false;
  while(!done)
  {
    X = exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / (double)std::sqrt(X);
    
    if(runif(0.0,1.0) <= gX) {
      done = true;
    }
  }
  
  return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t){
  double X, u;
  double mu = 1.0/z;
  
  // Pick sampler
  if(mu > t) {
    // Sampler based on truncated gamma 
    // Algorithm 3 in the Windle (2013) PhD thesis, page 128
    while(1) {
	  u = runif(0.0, 1.0);
      X = 1.0 / truncgamma();
      
	  if ((double)std::log(u) < (-z*z*0.5*X)) {
        break;
      }
    }
  }  
  else {
    // Rejection sampler
    X = t + 1.0;
    while(X >= t) {
      X = randinvg(mu);
    }
  }    
  return X;
}

// Sample PG(1,z)
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z){
  //  PG(b, z) = 0.25 * J*(b, z/2)
  z = (double)std::fabs((double)z) * 0.5;
  
  // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
  double t = MATH_2_PI;
  
  // Compute p, q and the ratio q / (q + p)
  // (derived from scratch; derivation is not in the original paper)
  double K = z*z/2.0 + MATH_PI2/8.0;
  double logA = (double)std::log(4.0) - MATH_LOG_PI - z;
  double logK = (double)std::log(K);
  double Kt = K * t;
  double w = (double)std::sqrt(MATH_PI_2);

  double logf1 = logA + pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z + pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = (double)std::exp(logf1) + (double)std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q); 

  double u, X;

  // Main sampling loop; page 130 of the Windle PhD thesis
  while(1) 
  {
    // Step 1: Sample X ? g(x|z)
    u = runif(0.0,1.0);
    if(u < ratio) {
      // truncated exponential
      X = t + exprnd(1.0)/K;
    }
    else {
      // truncated Inverse Gaussian
      X = tinvgauss(z, t);
    }

    // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
    int i = 1;
    double Sn = aterm(0, X, t);
    double U = runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;

    while(1) 
    {
      Sn = Sn + asgn * aterm(i, X, t);
      
      // Accept if n is odd
      if(!even && (U <= Sn)) {
        X = X * 0.25;
        return X;
      }
      
      // Return to step 1 if n is even
      if(even && (U > Sn)) {
        break;
      }
      
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  return X;
}

double rpg(int n, double z){
  
  double sum = 0;
  int i;
 
  for(i = 0; i < n; i++){
    sum += samplepg(z);
  }
 
  return(sum);
}
////

void show(int *a, int n);
void show(double *a, int n);
void show(int *a, int r, int c);
void show(double *a, int r, int c);
void zeros(double *a, int n);
void writeRMatrix(string outfile, double * a, int nrow, int ncol);
void writeRMatrix(string outfile, int * a, int nrow, int ncol);
void mvrnorm(double *des, double *mu, double *cholCov, int dim);
void covTransInv(double *z, double *v, int m);
void covTrans(double *v, double *z, int m);
void covTrans(vector<double> v, double *z, int m);
void covTransInvExpand(double *v, double *z, int m);
void covExpand(double *v, double *z, int m);
double logit(double theta, double a, double b);
double logitInv(double z, double a, double b);
double dist2(double &a1, double &a2, double &b1, double &b2);

//Description: sorts vectors a and b of length n based on decreasing values of a.
void fSort(double *a, int *b, int n);

//Description: given a location's index i and number of neighbors m this function provides the index to i and number of neighbors in nnIndx
void getNNIndx(int i, int m, int &iNNIndx, int &iNN);

//Description: creates the nearest neighbor index given pre-ordered location coordinates.
//Input:
//n = number of locations
//m = number of nearest neighbors
//coords = ordered coordinates for the n locations
//Output:
//nnIndx = set of nearest neighbors for all n locations (on return)
//nnDist = euclidean distance corresponding to nnIndx (on return)
//nnIndxLU = nx2 look-up matrix with row values correspond to each location's index in nnIndx and number of neighbors (columns 1 and 2, respectively)
//Note: nnIndx and nnDist must be of length (1+m)/2*m+(n-m-1)*m on input. nnIndxLU must also be allocated on input.
void mkNNIndx(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);

//void mkNNIndx2(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU);

//which index of b equals a, where b is of length n
int which(int a, int *b, int n){
  int i;
  for(i = 0; i < n; i++){
    if(a == b[i]){
      return(i);
    }
  }

  cout << "c++ error: which failed" << endl;
  return -9999;
}

//Description: using the fast mean-distance-ordered nn search by Ra and Kim 1993
//Input:
//ui = is the index for which we need the m nearest neighbors
//m = number of nearest neighbors
//n = number of observations, i.e., length of u
//sIndx = the NNGP ordering index of length n that is pre-sorted by u
//u = x+y vector of coordinates assumed sorted on input
//rSIndx = vector or pointer to a vector to store the resulting nn sIndx (this is at most length m for ui >= m)
//rNNDist = vector or point to a vector to store the resulting nn Euclidean distance (this is at most length m for ui >= m)  

double dmi(double *x, double *c, int inc){
    return pow(x[0]+x[inc]-c[0]-c[inc], 2);
}

double dei(double *x, double *c, int inc){
  return pow(x[0]-c[0],2)+pow(x[inc]-c[inc],2);
}

void fastNN(int m, int n, double *coords, int ui, double *u, int *sIndx, int *rSIndx, double *rSNNDist){
  
  int i,j,k;
  bool up, down;
  double dm, de;
  
  //rSNNDist will hold de (i.e., squared Euclidean distance) initially.
  for(i = 0; i < m; i++){
    rSNNDist[i] = std::numeric_limits<double>::infinity();
  }
  
  i = j = ui;
  
  up = down = true;
  
  while(up || down){
    
    if(i == 0){
      down = false;
    }

    if(j == (n-1)){
      up = false;
    }

    if(down){
      
      i--;
      
      dm = dmi(&coords[sIndx[ui]], &coords[sIndx[i]], n);
      
      if(dm > 2*rSNNDist[m-1]){
	down = false;
	
      }else{
	de = dei(&coords[sIndx[ui]], &coords[sIndx[i]], n);

	if(de < rSNNDist[m-1] && coords[sIndx[i]] < coords[sIndx[ui]]){
	  rSNNDist[m-1] = de;
	  rSIndx[m-1] = sIndx[i];
	  rsort_with_index(rSNNDist, rSIndx, m);
	}
	
      }
    }//end down
    
    if(up){
      
      j++;
      
      dm = dmi(&coords[sIndx[ui]], &coords[sIndx[j]], n);
      
      if(dm > 2*rSNNDist[m-1]){
	up = false;
	
      }else{
	de = dei(&coords[sIndx[ui]], &coords[sIndx[j]], n);

	if(de < rSNNDist[m-1] && coords[sIndx[j]] < coords[sIndx[ui]]){
	  rSNNDist[m-1] = de;
	  rSIndx[m-1] = sIndx[j];
	  rsort_with_index(rSNNDist, rSIndx, m);
	}
	
      }
      
    }//end up
    
  }
  
  for(i = 0; i < m; i++){
    rSNNDist[i] = sqrt(rSNNDist[i]);
  }


}


//Description: given the nnIndex this function fills uIndx for identifying those locations that have the i-th location as a neighbor.
//Input:
//n = number of locations
//m = number of nearest neighbors
//nnIndx = set of nearest neighbors for all n locations
//Output:
//uIndx = holds the indexes for locations that have each location as a neighbor
//uIndxLU = nx2 look-up matrix with row values correspond to each location's index in uIndx and number of neighbors (columns 1 and 2, respectively)
//Note: uIndx must be of length (1+m)/2*m+(n-m-1)*m on input. uINdxLU must also be allocated on input.
void mkUIndx(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU);

//Description: writes nnIndex to file with each row corresponding to the ordered location coordinates (using R 1 offset). Each row corresponds the coordinate's index followed by its nearest neighbor indexes.
void writeRNNIndx(string outfile, int *nnIndx, int n, int m);

//Description: same as the other writeRNNIndx but this one uses the nnIndxLU look-up (duplicated function just for testing).
void writeRNNIndx(string outfile, int *nnIndx, int *nnIndxLU, int n);

//Description: same as the other writeRNNIndx but this one uses the nnIndxLU look-up (duplicated function just for testing).
void writeRA(string outfile, double *A, int *nnIndxLU, int n);

//Description: same as writeRNNIndx but indexes in each row identify those locations that have the i-th row as a neighbor.
void writeRUIndx(string outfile, int *uIndx, int *uIndxLU, int n);

//Description: computes the quadratic term.
double Q(double *B, double *F, double *u, double *v, int n, int *nnIndx, int *nnIndxLU){
  
  double a, b, q = 0;
  int i, j;

#pragma omp parallel for private(a, b, j) reduction(+:q)
  for(i = 0; i < n; i++){
    a = 0;
    b = 0;
    for(j = 0; j < nnIndxLU[n+i]; j++){
      a += B[nnIndxLU[i]+j]*u[nnIndx[nnIndxLU[i]+j]];
      b += B[nnIndxLU[i]+j]*v[nnIndx[nnIndxLU[i]+j]];
    }
    q += (u[i] - a)*(v[i] - b)/F[i];
  }
  
  return(q);
}

//Description: update B and F.
void updateBF(double *B, double *F, double *c, double *C, double *D, double *d, int *nnIndxLU, int *CIndx, int n, double sigmaSq, double phi){
    
  int i, k, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';
  double logDet = 0;
    
#pragma omp parallel for private(k, l)
    for(i = 0; i < n; i++){
      if(i > 0){
	for(k = 0; k < nnIndxLU[n+i]; k++){
	  c[nnIndxLU[i]+k] = sigmaSq*exp(-phi*d[nnIndxLU[i]+k]);
	  for(l = 0; l <= k; l++){
	    C[CIndx[i]+l*nnIndxLU[n+i]+k] = sigmaSq*exp(-phi*D[CIndx[i]+l*nnIndxLU[n+i]+k]);
	  }
	}
	dpotrf_(&lower, &nnIndxLU[n+i], &C[CIndx[i]], &nnIndxLU[n+i], &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
	dpotri_(&lower, &nnIndxLU[n+i], &C[CIndx[i]], &nnIndxLU[n+i], &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
	dsymv_(&lower, &nnIndxLU[n+i], &one, &C[CIndx[i]], &nnIndxLU[n+i], &c[nnIndxLU[i]], &inc, &zero, &B[nnIndxLU[i]], &inc);
	F[i] = sigmaSq - ddot_(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[nnIndxLU[i]], &inc);
      }else{
	B[i] = 0;
	F[i] = sigmaSq;
      }
    }
   
}

void dimCheck(string test, int i, int j){
  if(i != j)
    cout << test << " " << i << "!=" << j << endl;
}


int main(int argc, char **argv){
  int i, j, k, l, s;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double negOne = -1.0;
  double zero = 0.0;
  char lower = 'L';
  char upper = 'U';
  char ntran = 'N';
  char ytran = 'T';
  char rside = 'R';
  char lside = 'L';
  
  string parfile;
  if(argc > 1)
    parfile = argv[1];
  else
    parfile = "pfile";
  
  kvpar par(parfile);
  
  bool debug = false;
  
  //Get stuff
  int nnIndexOnly; par.getVal("nn.index.only", nnIndexOnly);
  int nnFast; par.getVal("nn.fast", nnFast);
  int nThreads; par.getVal("n.threads", nThreads);
  int seed; par.getVal("seed", seed);
  int nSamples; par.getVal("n.samples", nSamples);
  int nSamplesStart; par.getVal("n.samples.start", nSamplesStart);
  int nReport; par.getVal("n.report", nReport);
  string outFile; par.getVal("out.file", outFile);
    
  omp_set_num_threads(nThreads);
  
  //set seed
  set_seed(123,seed);
  
  //m number of nearest neighbors
  //n number of locations
  //p number of columns of X, later P = p*h (code assumes same number of covars for each outcome)
  //q columns of Lambda, i.e., number of GPs
  //h number of outcomes at each location
  
  int m; par.getVal("m", m);
  int n; par.getVal("n", n);
  int p; par.getVal("p", p);
  int q; par.getVal("q", q);
  int h; par.getVal("h", h);
  int P = p*h;
  int nq = n*q;
  int qq = q*q;
  int pp = p*p;
  int hq = h*q;
  int np = n*p;
  int nh = n*h;
  
  //data and starting values and some dim checks
  double *X = NULL; par.getFile("X.file", X, i, j); //p x nq (stacked by outcome, simplifies life below).
  double *z = NULL; par.getFile("z.file", z, i, j); //nq x 1 (stacked by location).
  int *missing = NULL; par.getFile("y.missing.file", missing, i, j); //nh x 1 (stacked by location).
  double *coords = NULL; par.getFile("coords.file", coords, i, j);
  double *lambda = NULL; par.getFile("lambda.starting.file", lambda, i, j); 
  double *beta = NULL; par.getFile("beta.starting.file", beta, i, j);
  //double *tauSq = NULL; par.getFile("tauSq.starting.file", tauSq, i, j);
  double *phi = NULL; par.getFile("phi.starting.file", phi, i, j);
  double *w = NULL; par.getFile("w.starting.file", w, i, j);
    
  //priors and tuning
  double phi_a; par.getVal("phi.a", phi_a);
  double phi_b; par.getVal("phi.b", phi_b);
  vector<double>  phiTuning; par.getVal("phi.tuning", phiTuning);
  
  //double tauSq_a = 2.0;
  //double *tauSq_b = NULL; par.getFile("tauSq.b.file", tauSq_b, i, j);
   
  //allocated for the nearest neighbor index vector (note, first location has no neighbors).
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
  int *nnIndx = new int[nIndx];
  double *d = new double[nIndx];
  int *nnIndxLU = new int[2*n]; //first column holds the nnIndx index for the i-th location and the second columns holds the number of neighbors the i-th location has (the second column is a bit of a waste but will simplifying some parallelization).
  
  //make the neighbor index
  if(!nnFast){

    mkNNIndx(n, m, coords, nnIndx, d, nnIndxLU);
    //writeRNNIndx("nnIndx-slow", nnIndx, n, m);
    
  }else{
    
    int *sIndx = new int[n];
    double *u = new double[n];
    
    for(i = 0; i < n; i++){
      sIndx[i] = i;
      u[i] = coords[i]+coords[n+i];
    }
    
    cout << "sort" << endl;
    rsort_with_index(u, sIndx, n); 
    
    int iNNIndx, iNN;
    
    cout << "fastNN" << endl;
    //make nnIndxLU and fill nnIndx and d
#pragma omp parallel for private(iNNIndx, iNN)
    for(i = 0; i < n; i++){ //note this i indexes the u vector
      getNNIndx(sIndx[i], m, iNNIndx, iNN);
      nnIndxLU[sIndx[i]] = iNNIndx;
      nnIndxLU[n+sIndx[i]] = iNN;   
      fastNN(iNN, n, coords, i, u, sIndx, &nnIndx[iNNIndx], &d[iNNIndx]);
    } 
    //writeRNNIndx("nnIndx-fast", nnIndx, n, m);
  }
  
  if(nnIndexOnly){
    exit(1);
  }

  //allocate for the U index vector that keep track of which locations have the i-th location as a neighbor
  int *uIndx = new int[nIndx]; //U indexes 

  //first column holds the uIndx index for i-th location and second column holds
  //the number of neighbors for which the i-th location is a neighbor
  int *uIndxLU = new int[2*n]; 
  
  //make u index
  mkUIndx(n, m, nnIndx, uIndx, uIndxLU);

  //u lists those locations that have the i-th location as a neighbor
  //then for each of those locations that have i as a neighbor, we need to know the index of i in each of their B vectors (i.e. where does i fall in their neighbor set)
  int *uiIndx = new int[nIndx];

  for(i = 0; i < n; i++){//for each i
    for(j = 0; j < uIndxLU[n+i]; j++){//for each location that has i as a neighbor
      k = uIndx[uIndxLU[i]+j];//index of a location that has i as a neighbor
      uiIndx[uIndxLU[i]+j] = which(i, &nnIndx[nnIndxLU[k]], nnIndxLU[n+k]);
    }
  }

  //PG augmentation stuff.
  double *omega = new double[nh];
  double *kappa = new double[nh];
  double *yStr = new double[nh];
  double *nTrial = new double[nh];// Might have some real binomial data eventually but for now ...
  for(i = 0; i < nh; i++){
    nTrial[i] = 1.0;
  }
  
  for(i = 0; i < nh; i++){
    kappa[i] = z[i] - nTrial[i]/2.0;
  }  

  //return stuff
  int nSamplesKeep = (nSamples-nSamplesStart);
  double *betaSamples = new double[P*nSamplesKeep];
  double *phiSamples = new double[q*nSamplesKeep];
  //double *tauSqSamples = new double[h*nSamplesKeep];
  double *wSamples = new double[nq*nSamplesKeep];
  double *lambdaSamples = new double[hq*nSamplesKeep];
  double *fittedSamples = new double[nh*nSamplesKeep];
  double *ZwSamples = new double[nh*nSamplesKeep];
 
  //other stuff
  double **B = new double*[q];
  double **F = new double*[q];
  for(i = 0; i < q; i++){
    B[i] = new double[nIndx];
    F[i] = new double[n];
  }
  
  double *c = new double[nIndx];

  int *CIndx = new int[2*n]; //index for D and C.
  for(i = 0, j = 0; i < n; i++){//zero should never be accessed
    j += nnIndxLU[n+i]*nnIndxLU[n+i];
    if(i == 0){
      CIndx[n+i] = 0;
      CIndx[i] = 0;
    }else{
      CIndx[n+i] = nnIndxLU[n+i]*nnIndxLU[n+i]; 
      CIndx[i] = CIndx[n+i-1] + CIndx[i-1];
    }
  }
 
  double *C = new double[j]; zeros(C, j);
  double *D = new double[j]; zeros(D, j);

  for(i = 0; i < n; i++){
    for(k = 0; k < nnIndxLU[n+i]; k++){   
      for(l = 0; l <= k; l++){
  	D[CIndx[i]+l*nnIndxLU[n+i]+k] = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
      }
    }
  }

  if(debug){
    for(i = 0; i < n; i++){
      show(&d[nnIndxLU[i]], 1, nnIndxLU[n+i]);
      cout << "--" << endl;
      show(&D[CIndx[i]], nnIndxLU[n+i], nnIndxLU[n+i]);
      cout << "-------" << endl;
    }
  }
  
  //other stuff
  double logPostCand, logPostCurrent, status = 0;
  double *accept = new double[q]; zeros(accept, q);
  double *batchAccept = new double[q]; zeros(batchAccept, q);
  double *phiCand = new double[q];

  //for gibbs update of beta's
  double **XtX = new double*[h];
  
  for(i = 0, j = 0; i < h; i++){
    XtX[i] = new double[pp];
    //dgemm_(&ntran, &ytran, &p, &p, &n, &one, &X[j], &p, &X[j], &p, &zero, XtX[i], &p);
    //j += np;
  }

  double *tmp_nh = new double[nh];
  double *tmp_nh2 = new double[nh];
  double *tmp_nq = new double[nq];
  double *tmp_h = new double[h];
  double *tmp_h2 = new double[h];
  double *tmp_hq = new double[hq];
  double *tmp_qq = new double[qq];
  double *tmp_qq2 = new double[qq];
  double *tmp_q = new double[q];
  double *tmp_q2 = new double[q];
  double *tmp_p = new double[p];
  double *tmp_p2 = new double[p];
  double *tmp_pp = new double[pp];
  double *tmp_n = new double[n];
  double *tmp_pn = new double[np];
  double *mu = new double[q];
  double *var = new double[qq];
  double *a = new double[q];
  double *v = new double[q];
  double *f = new double[q];
  double *g = new double[q];
  double aa, b, e, aij, logDet;
  int jj, kk, ll;

  double wall0 = get_wall_time();
  double cpu0 = get_cpu_time();
  double wall1, cpu1;
  int sKeep = 0;
  
  cout << "start sampling" << endl;
  
  for(s = 0; s < nSamples; s++){
    
    ///////////////////////////////
    //Update missing y and PG augs.
    ///////////////////////////////

    //#pragma omp parallel for
    for(i = 0; i < h; i++){
      dgemv_(&ytran, &p, &n, &one, &X[i*np], &p, &beta[i*p], &inc, &zero, &tmp_nh[i], &h);
    }

    //for each location [location 1 hx1 | location 2 hx1 | ... | location n hx1]
    //#pragma omp parallel for
    for(i = 0; i < n; i++){
      dgemv_(&ntran, &h, &q, &one, lambda, &h, &w[i*q], &inc, &one, &tmp_nh[i*h], &inc);
    }

    for(i = 0; i < nh; i++){
      if(missing[i] == 1){
    	kappa[i] = rbinom(1.0, 1.0/(1.0 + exp(-1.0*tmp_nh[i]))) - nTrial[i]/2.0;
      }
    }

    for(i = 0; i < nh; i++){
      omega[i] = rpg(nTrial[i], tmp_nh[i]);
      yStr[i] = kappa[i]/omega[i];
    }

  
    ///////////////
    //update beta 
    ///////////////

    //This is confusing because omega is stacked by location like y, but X is pxnh stacked sideways by outcome.
    for(i = 0, j = 0; i < h; i++){
      
      for(k = 0; k < n; k++){
    	for(l = 0; l < p; l++){
    	  tmp_pn[p*k+l] = X[j+p*k+l]*omega[k*h+i];
    	}
      }
      
      dgemm_(&ntran, &ytran, &p, &p, &n, &one, tmp_pn, &p, &X[j], &p, &zero, XtX[i], &p);
      j += np;
    }

    for(i = 0; i < h; i++){

      // for(j = 0; j < n; j++){
      // 	tmp_n[j] = (z[j*h+i] - ddot_(&q, &w[j*q], &inc, &lambda[i], &h))/tauSq[i];
      // }
      // dgemv_(&ntran, &p, &n, &one, &X[np*i], &p, tmp_n, &inc, &zero, tmp_p, &inc); 
      
      // for(j = 0; j < pp; j++){
      // 	tmp_pp[j] = XtX[i][j]/tauSq[i];
      // }

      for(j = 0; j < n; j++){
    	tmp_n[j] = (yStr[j*h+i] - ddot_(&q, &w[j*q], &inc, &lambda[i], &h))*omega[j*h+i];///tauSq[i];
      }
      dgemv_(&ntran, &p, &n, &one, &X[np*i], &p, tmp_n, &inc, &zero, tmp_p, &inc); 
      
      for(j = 0; j < pp; j++){
    	tmp_pp[j] = XtX[i][j];//tauSq[i];
      }
    
      dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      dpotri_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
      
      dsymv_(&lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc);
      
      dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      
      mvrnorm(&beta[i*p], tmp_p2, tmp_pp, p);
    }
    
    
    ///////////////
    //update w 
    ///////////////
    
    //ll index the process, i.e., phi_ll ll = 1,2,...,q
    for(ll = 0; ll < q; ll++){     
      updateBF(B[ll], F[ll], c, C, D, d, nnIndxLU, CIndx, n, one, phi[ll]);
    }

    //for each location
    for(i = 0; i < n; i++){
      
      //tmp_qq = lambda' Psi^{-1} lambda
      for(k = 0; k < h; k++){
    	for(j = 0; j < q; j++){
    	  tmp_hq[j*h+k] = lambda[j*h+k]*omega[i*h+k];
    	}
      }
      
      dgemm_(&ytran, &ntran, &q, &q, &h, &one, tmp_hq, &h, lambda, &h, &zero, tmp_qq, &q);
      
      for(ll = 0; ll < q; ll++){
	
    	a[ll] = 0;
    	v[ll] = 0;
	
    	if(uIndxLU[n+i] > 0){//is i a neighbor for anybody
	  
    	  for(j = 0; j < uIndxLU[n+i]; j++){//how many location have i as a neighbor
    	    b = 0;
	    
    	    //now the neighbors for the jth location who has i as a neighbor
    	    jj = uIndx[uIndxLU[i]+j]; //jj is the index of the jth location who has i as a neighbor
	    
    	    for(k = 0; k < nnIndxLU[n+jj]; k++){// these are the neighbors of the jjth location
    	      kk = nnIndx[nnIndxLU[jj]+k];// kk is the index for the jth locations neighbors
	      
    	      if(kk != i){//if the neighbor of jj is not i
    		b += B[ll][nnIndxLU[jj]+k]*w[kk*q+ll];//covariance between jj and kk and the random effect of kk
    	      }
    	    }
	    
    	    aij = w[jj*q+ll] - b;
	    	    
    	    a[ll] += B[ll][nnIndxLU[jj]+uiIndx[uIndxLU[i]+j]]*aij/F[ll][jj];
    	    v[ll] += pow(B[ll][nnIndxLU[jj]+uiIndx[uIndxLU[i]+j]],2)/F[ll][jj];
	    
    	  }
	  
    	}

    	e = 0;
    	for(j = 0; j < nnIndxLU[n+i]; j++){
    	  e += B[ll][nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]*q+ll];
    	}
	
    	f[ll] = 1.0/F[ll][i];
    	g[ll] = e/F[ll][i];
	
      }//end q
      
      //var
      dcopy_(&qq, tmp_qq, &inc, var, &inc);
      for(j = 0; j < q; j++){
      	var[j*q+j] += f[j] + v[j];
      }
      dpotrf_(&lower, &q, var, &q, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      dpotri_(&lower, &q, var, &q, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
      
      //mu
      for(j = 0; j < h; j++){
      	tmp_h[j] = (yStr[i*h+j] - ddot_(&p, &X[np*j+i*p], &inc, &beta[j*p], &inc))*omega[i*h+j];///tauSq[j];
      }
      
      dgemv_(&ytran, &h, &q, &one, lambda, &h, tmp_h, &inc, &zero, mu, &inc);
      
      for(j = 0; j < q; j++){
      	mu[j] += g[j] + a[j];
      }
      
      dsymv_(&lower, &q, &one, var, &q, mu, &inc, &zero, tmp_h, &inc);
      
      //draw
      dpotrf_(&lower, &q, var, &q, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}   

      mvrnorm(&w[i*q], tmp_h, var, q);
   
    }//end n


    ///////////////
    //update lambda 
    ///////////////
    
    for(i = 1; i < h; i++){

      zeros(tmp_q, q);
      zeros(tmp_qq, qq);
      zeros(tmp_qq2, qq);
      
      for(k = 0; k < q; k++){
    	for(l = 0; l < q; l++){
    	  for(j = 0; j < n; j++){
    	    tmp_qq[k*q+l] += w[j*q+k]*w[j*q+l]*omega[j*h+i];
    	  }
    	}
      }
      
      if(i < q){//ll will be the dim of mu and var
    	ll = i;
      }else{
    	ll = q;
      }	
      
      //mu
      for(j = 0; j < n; j++){
    	tmp_n[j] = yStr[j*h+i] - ddot_(&p, &X[i*np+j*p], &inc, &beta[i*p], &inc);
	
    	if(i < q){
    	  tmp_n[j] -= w[j*q+i];
    	}
      }

      for(j = 0, l = 0; j < n; j++){
    	for(k = 0; k < q; k++, l++){
    	  tmp_nq[l] = omega[j*h+i]*w[j*q+k];
    	}
      }

      for(k = 0; k < ll; k++){
    	for(j = 0; j < n; j++){
    	  tmp_q[k] += tmp_nq[j*q+k]*tmp_n[j];
    	}
      }
      
      // for(k = 0; k < ll; k++){
      // 	for(j = 0; j < n; j++){
      // 	  tmp_q[k] += w[j*q+k]*tmp_n[j];
      // 	}
      // }

      // for(k = 0; k < ll; k++){
      // 	tmp_q[k] /= tauSq[i];
      // }     

      //var      
      for(k = 0, l = 0; k < ll; k++){
    	for(j = 0; j < ll; j++, l++){
    	  tmp_qq2[l] = tmp_qq[k*q+j];
    	}
      }
      
      // for(j = 0; j < ll*ll; j++){
      // 	tmp_qq2[j] /= tauSq[i];
      // }
      
      for(j = 0; j < ll; j++){
    	tmp_qq2[j*ll+j] += 1.0;
      }

      dpotrf_(&lower, &ll, tmp_qq2, &ll, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      dpotri_(&lower, &ll, tmp_qq2, &ll, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
      
      dsymv_(&lower, &ll, &one, tmp_qq2, &ll, tmp_q, &inc, &zero, tmp_q2, &inc);
      
      dpotrf_(&lower, &ll, tmp_qq2, &ll, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      
      mvrnorm(tmp_q, tmp_q2, tmp_qq2, ll);
      dcopy_(&ll, tmp_q, &inc, &lambda[i], &h);
    }

//     /////////////////////
//     //update tau^2
//     /////////////////////
    
//     //Note, X is pxnh stacked by outcome [outcome 1 nx1 | outcome 2 nx1 | ... | outcome h nx1],
//     //I wanted the output of X'beta to be stacked by location, i.e.,
//     //[location 1 hx1 | location 2 hx1 | ... | location n hx1] to match the vector y
// #pragma omp parallel for
//     for(i = 0; i < h; i++){
//       dgemv_(&ytran, &p, &n, &one, &X[i*np], &p, &beta[i*p], &inc, &zero, &tmp_nh[i], &h);
//     }
    
//     //for each location [location 1 hx1 | location 2 hx1 | ... | location n hx1]
// #pragma omp parallel for
//     for(i = 0; i < n; i++){
//       dgemv_(&ntran, &h, &q, &one, lambda, &h, &w[i*q], &inc, &one, &tmp_nh[i*h], &inc);
//     }
    
//     for(i = 0; i < nh; i++){
//       tmp_nh[i] = z[i] - tmp_nh[i];
//     }

//     //Note, I'm just used a standard IG (because I'm lazy), but we should experiment with the half-t spelled out in the Taylor-Rodriguez appendix.
//     for(i = 0; i < h; i++){
//       tauSq[i] = 1.0/rgamma(tauSq_a+n/2.0, 1.0/(tauSq_b[i]+0.5*ddot_(&n, &tmp_nh[i], &h, &tmp_nh[i], &h)));
//     }
    
    ///////////////
    //update phi
    ///////////////

    for(ll = 0; ll < q; ll++){
      
      //current   
      updateBF(B[ll], F[ll], c, C, D, d, nnIndxLU, CIndx, n, one, phi[ll]);
      
      aa = 0;
      logDet = 0;

#pragma omp parallel for private (e, j, b) reduction(+:aa, logDet)
      for(i = 0; i < n; i++){
	
	if(nnIndxLU[n+i] > 0){
	  
	  e = 0;
	  
	  for(j = 0; j < nnIndxLU[n+i]; j++){
	    e += B[ll][nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]*q+ll];
	  }
	  
	  b = w[i*q+ll] - e;
	  
	}else{
	  
	  b = w[i*q+ll];
	  
	}
	
	aa += b*b/F[ll][i];
	logDet += log(F[ll][i]);
      }
      
      logPostCurrent = -0.5*logDet - 0.5*aa;
      logPostCurrent += log(phi[ll] - phi_a) + log(phi_b - phi[ll]); 
      
      //candidate
      phiCand[ll] = logitInv(rnorm(logit(phi[ll], phi_a, phi_b), phiTuning[ll]), phi_a, phi_b);
      
      updateBF(B[ll], F[ll], c, C, D, d, nnIndxLU, CIndx, n, one, phiCand[ll]);
      
      aa = 0;
      logDet = 0;

#pragma omp parallel for private (e, j, b) reduction(+:aa, logDet)
      for(i = 0; i < n; i++){
	
	if(nnIndxLU[n+i] > 0){
	  
	  e = 0;
	  
	  for(j = 0; j < nnIndxLU[n+i]; j++){
	    e += B[ll][nnIndxLU[i]+j]*w[nnIndx[nnIndxLU[i]+j]*q+ll];
	  }
	  
	  b = w[i*q+ll] - e;
	  
	}else{
	  
	  b = w[i*q+ll];
	  
	}
	
	aa += b*b/F[ll][i];
	logDet += log(F[ll][i]);
      }
      
      logPostCand = -0.5*logDet - 0.5*aa;
      logPostCand += log(phiCand[ll] - phi_a) + log(phi_b - phiCand[ll]); 
      
      if(runif(0.0,1.0) <= exp(logPostCand - logPostCurrent)){
	phi[ll] = phiCand[ll];
	accept[ll]++;
	batchAccept[ll]++;	
      }
      
    }//end q


    /////////////////////////////////////////////
    //fit 
    /////////////////////////////////////////////
    
    //Note, X is pxnh stacked by outcome [outcome 1 nx1 | outcome 2 nx1 | ... | outcome h nx1],
    //I wanted the output of X'beta to be stacked by location, i.e.,
    //[location 1 hx1 | location 2 hx1 | ... | location n hx1] to match the vector y
#pragma omp parallel for
   for(i = 0; i < h; i++){
     dgemv_(&ytran, &p, &n, &one, &X[i*np], &p, &beta[i*p], &inc, &zero, &tmp_nh[i], &h);
   }

   //for each location [location 1 hx1 | location 2 hx1 | ... | location n hx1]
#pragma omp parallel for
   for(i = 0; i < n; i++){
     dgemv_(&ntran, &h, &q, &one, lambda, &h, &w[i*q], &inc, &one, &tmp_nh[i*h], &inc);
     dgemv_(&ntran, &h, &q, &one, lambda, &h, &w[i*q], &inc, &zero, &tmp_nh2[i*h], &inc);
    }

   for(i = 0; i < nh; i++){
     tmp_nh[i] = 1.0/(1.0 + exp(-1.0*tmp_nh[i]));
   }

    if(s >= nSamplesStart){
      dcopy_(&P, beta, &inc, &betaSamples[sKeep*P], &inc);
      dcopy_(&hq, lambda, &inc, &lambdaSamples[sKeep*hq], &inc);
      //dcopy_(&h, tauSq, &inc, &tauSqSamples[sKeep*h], &inc);
      dcopy_(&q, phi, &inc, &phiSamples[sKeep*q], &inc);
      dcopy_(&nq, w, &inc, &wSamples[sKeep*nq], &inc);
      dcopy_(&nh, tmp_nh2, &inc, &ZwSamples[sKeep*nh], &inc);
      dcopy_(&nh, tmp_nh, &inc, &fittedSamples[sKeep*nh], &inc);
      sKeep++;
    }
    
    /////////////////////////////////////////////
    //report
    /////////////////////////////////////////////
    
    if(status == nReport){
      cout << "percent complete: " << 100*s/nSamples << endl;   
      cout << "phi acceptance:" << endl;   
      for(i = 0; i < q; i++){
    	cout << "phi " << i << ": " << 100.0*batchAccept[i]/nReport << endl;
    	batchAccept[i] = 0;
      }
      cout << "---------------" << endl;
      status = 0;

      //timer
      wall1 = get_wall_time();
      cpu1  = get_cpu_time();
      
      cout << "Wall Time = " << wall1 - wall0 << endl;
      cout << "CPU Time  = " << cpu1  - cpu0  << endl;
      
      wall0 = get_wall_time();
      cpu0 = get_cpu_time();
      
    }
    status++;
    
  }
			 
  writeRMatrix(outFile+"-beta", betaSamples, P, nSamplesKeep);
  writeRMatrix(outFile+"-phi", phiSamples, q, nSamplesKeep);
  //writeRMatrix(outFile+"-tauSq", tauSqSamples, h, nSamplesKeep);
  writeRMatrix(outFile+"-w", wSamples, nq, nSamplesKeep);
  writeRMatrix(outFile+"-Zw", ZwSamples, nh, nSamplesKeep);
  writeRMatrix(outFile+"-lambda", lambdaSamples, hq, nSamplesKeep);
  writeRMatrix(outFile+"-fitted", fittedSamples, nh, nSamplesKeep); 

  return(0);
}

void writeRMatrix(string outfile, double * a, int nrow, int ncol){

    ofstream file(outfile.c_str());
    if ( !file ) {
      cerr << "Data file could not be opened." << endl;
      exit(1);
    }
    
    for(int i = 0; i < nrow; i++){
      for(int j = 0; j < ncol-1; j++){
	file << setprecision(10) << fixed << a[j*nrow+i] << "\t";
      }
      file << setprecision(10) << fixed << a[(ncol-1)*nrow+i] << endl;    
    }
    file.close();
}


void writeRMatrix(string outfile, int* a, int nrow, int ncol){
  
  ofstream file(outfile.c_str());
  if ( !file ) {
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }
  
  
  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol-1; j++){
      file << fixed << a[j*nrow+i] << "\t";
    }
    file << fixed << a[(ncol-1)*nrow+i] << endl;    
  }
  file.close();
}

void mvrnorm(double *des, double *mu, double *cholCov, int dim){
  
  int i;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  
  for(i = 0; i < dim; i++){
    des[i] = rnorm(0, 1);
  }
 
  dtrmv_("L", "N", "N", &dim, cholCov, &dim, des, &inc);
  daxpy_(&dim, &one, mu, &inc, des, &inc);
}

void show(double *a, int n){
  for(int i = 0; i < n; i++)
    cout << setprecision(20) << fixed << a[i] << endl;
}


void show(int *a, int n){
  for(int i = 0; i < n; i++)
    cout << fixed << a[i] << endl;
}


void zeros(double *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 0.0;
}


void show(double *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){
      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}


void show(int *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){

      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}

void covTransInv(double *z, double *v, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      v[k] = z[k];
      if(i == j)
	v[k] = exp(z[k]);
    }
  }

}

void covTrans(double *v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTrans(vector<double> v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTransInvExpand(double *v, double *z, int m){
  int i, j, k;
  
  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
      if(i == j)
	z[i*m+j] = exp(z[i*m+j]);
    }
  }
  
}

void covExpand(double *v, double *z, int m){
  int i, j, k;
  
  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
    }
  }
  
}

double logit(double theta, double a, double b){
  return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
  return b-(b-a)/(1+exp(z));
}

double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}

void fSort(double *a, int *b, int n){
  
  int j, k, l;
  double v;
  
  for(j = 1; j <= n-1; j++){
    k = j;  
    while(k > 0 && a[k] < a[k-1]) {
      v = a[k]; l = b[k];
      a[k] = a[k-1]; b[k] = b[k-1];
      a[k-1] = v; b[k-1] = l;
      k--;
    }
  }
}



void getNNIndx(int i, int m, int &iNNIndx, int &iNN){
  
  if(i == 0){
    iNNIndx = 0;//this should never be accessed
    iNN = 0;
    return;
  }else if(i < m){
    iNNIndx = static_cast<int>(static_cast<double>(i)/2*(i-1));
    iNN = i;
    return;
  }else{
    iNNIndx = static_cast<int>(static_cast<double>(m)/2*(m-1)+(i-m)*m);
    iNN = m;
    return;
  } 
}


void mkNNIndx(int n, int m, double *coords, int *nnIndx, double *nnDist, int *nnIndxLU){
  
  int i, j, iNNIndx, iNN;
  double d;
  
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
  
  for(i = 0; i < nIndx; i++){
    nnDist[i] = std::numeric_limits<double>::infinity();
  }
  
  #pragma omp parallel for private(j, iNNIndx, iNN, d)
  for(i = 0; i < n; i++){ 
    getNNIndx(i, m, iNNIndx, iNN);
    nnIndxLU[i] = iNNIndx;
    nnIndxLU[n+i] = iNN;   
    if(i != 0){  
      for(j = 0; j < i; j++){	
	d = dist2(coords[i], coords[n+i], coords[j], coords[n+j]);	
	if(d < nnDist[iNNIndx+iNN-1]){	  
	  nnDist[iNNIndx+iNN-1] = d;
	  nnIndx[iNNIndx+iNN-1] = j;
	  fSort(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN); 	  
	}	
      }
    } 
  }
  
}


void writeRNNIndx(string outfile, int *nnIndx, int n, int m){
  
  ofstream file(outfile.c_str());
  if(!file){
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }
  
  int i, j, a, b;
  
  for(i = 0; i < n; i++){
    if(i != 0){    
      getNNIndx(i, m, a, b);    
      file << i+1 << " ";
      for(j = 0; j < b; j++){
	if(j+1 == b){
	  file << nnIndx[a+j]+1;
	}else{
	  file << nnIndx[a+j]+1 << ",";
	}
      }
      file << endl;
    }
  }
  file.close();
}

void writeRNNIndx(string outfile, int *nnIndx, int *nnIndxLU, int n){
  
  ofstream file(outfile.c_str());
  if(!file){
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }
 
  int i, j;
  
  for(i = 0; i < n; i++){
    if(nnIndxLU[n+i] > 0){//i.e., not i = 0
      file << i+1 << " ";
      for(j = 0; j < nnIndxLU[n+i]; j++){
	if(j+1 == nnIndxLU[n+i]){
	  file << nnIndx[nnIndxLU[i]+j]+1;
	}else{	
	  file << nnIndx[nnIndxLU[i]+j]+1 << ",";	
	}
      }
      file << endl;
    }
  }
  
  file.close();
}

void writeRA(string outfile, double *A, int *nnIndxLU, int n){
  
  ofstream file(outfile.c_str());
  if(!file){
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }
 
  int i, j;
  
  for(i = 0; i < n; i++){
    if(nnIndxLU[n+i] > 0){//i.e., not i = 0
      file << i+1 << " ";
      for(j = 0; j < nnIndxLU[n+i]; j++){
	if(j+1 == nnIndxLU[n+i]){
	  file << A[nnIndxLU[i]+j];
	}else{	
	  file << A[nnIndxLU[i]+j] << ",";	
	}
      }
      file << endl;
    }
  }
  
  file.close();
}



void writeRUIndx(string outfile, int *uIndx, int *uIndxLU, int n){
  
  ofstream file(outfile.c_str());
  if(!file){
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }

  int i, j;
  
  for(i = 0; i < n; i++){
    file << i+1 << " ";
    if( uIndxLU[n+i] == 0){
  file << "NA";
}else{
    for(j = 0; j < uIndxLU[n+i]; j++){
      if(j+1 == uIndxLU[n+i]){
	file << uIndx[uIndxLU[i]+j]+1;
      }else{	
	file << uIndx[uIndxLU[i]+j]+1 << ",";	
      }
    }
}
    file << endl;
  }
  
  file.close();
}

void mkUIndx(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU){ 
  
  int iNNIndx, iNN, i, j, k, l, h;
  
  for(i = 0, l = 0; i < n; i++){    
    uIndxLU[i] = l; 
    for(j = 0, h = 0; j < n; j++){   
      getNNIndx(j, m, iNNIndx, iNN);  
      for(k = 0; k < iNN; k++){      	
	if(nnIndx[iNNIndx+k] == i){
	  uIndx[l+h] = j;
	  h++;
	}    
      }
    }
    l += h;
    uIndxLU[n+i] = h; 
  }
}



// void mkUIndx(int n, int m, int* nnIndx, int* uIndx, int* uIndxLU){ 
  
//   int iNNIndx, iNN, i, j, k, l, h;
  
//   for(i = 0; i < 2*n; i++){
//     uIndxLU[i] = 0;
//   }
  
// #pragma omp parallel for private(j, iNNIndx, iNN)
//   for(i = 0; i < n; i++){       
//     getNNIndx(i, m, iNNIndx, iNN);  
//     for(k = 0; k < iNN; k++){      	
//       if(nnIndx[iNNIndx+k] == i){
// 	uIndxLU[n+i]++;
//       }    
//     }
//   }


//   for(i = 1; i < n; i++){
//     uIndxLU[i] = uIndxLU[i-1]+uIndxLU[n+(i-1)];
//   }
  
// #pragma omp parallel for private(l, j, iNNIndx, iNN, k)
//   for(i = 0; i < n; i++){
//     l = uIndxLU[i];
//     for(j = 0; j < n; j++){
//       getNNIndx(j, m, iNNIndx, iNN);  
//       for(k = 0; k < iNN; k++){ 
// 	if(nnIndx[iNNIndx+k] == i){
// 	  uIndx[l] = j;
// 	  l++;
// 	  //break;
// 	}  
//       }
//     }
//   }

// }
