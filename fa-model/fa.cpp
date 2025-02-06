#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
using namespace std;

#include "../libs/kvpar.h"

#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <Rinternals.h>

#include <time.h>
#include <sys/time.h>

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
  int nThreads; par.getVal("n.threads", nThreads);
  int seed; par.getVal("seed", seed);
  int nSamples; par.getVal("n.samples", nSamples);
  int nSamplesStart; par.getVal("n.samples.start", nSamplesStart);
  int nReport; par.getVal("n.report", nReport);
  string outFile; par.getVal("out.file", outFile);
    
  omp_set_num_threads(nThreads);
  
  //set seed
  set_seed(123,seed);
  
  //n number of locations
  //p number of columns of X, later P = p*h (code assumes same number of covars for each outcome)
  //q columns of Lambda, i.e., number of GPs
  //h number of outcomes at each location
  
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
  double *X = NULL; par.getFile("X.file", X, i, j);
  double *z = NULL; par.getFile("z.file", z, i, j);
  double *lambda = NULL; par.getFile("lambda.starting.file", lambda, i, j); 
  double *beta = NULL; par.getFile("beta.starting.file", beta, i, j);
  double *tauSq = NULL; par.getFile("tauSq.starting.file", tauSq, i, j);
  double *w = NULL; par.getFile("w.starting.file", w, i, j);
    
  //priors and tuning
  double tauSq_a = 2.0;
  double *tauSq_b = NULL; par.getFile("tauSq.b.file", tauSq_b, i, j);
   
  //return stuff
  int nSamplesKeep = (nSamples-nSamplesStart);
  double *betaSamples = new double[P*nSamplesKeep];
  double *tauSqSamples = new double[h*nSamplesKeep];
  double *wSamples = new double[nq*nSamplesKeep];
  double *lambdaSamples = new double[hq*nSamplesKeep];
  double *fittedSamples = new double[nh*nSamplesKeep];

  //for gibbs update of beta's
  double **XtX = new double*[h];
  
  for(i = 0, j = 0; i < h; i++){
    XtX[i] = new double[p*p];
    dgemm_(&ntran, &ytran, &p, &p, &n, &one, &X[j], &p, &X[j], &p, &zero, XtX[i], &p);
    j += np;
  }

  double *tmp_nh = new double[nh];
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
  double *mu = new double[q];
  double *var = new double[qq];
  double *cholVar = new double[qq];

  int jj, kk, ll;
  int status = 0;
  int sKeep = 0;
  
  cout << "start sampling" << endl;
  
  for(s = 0; s < nSamples; s++){
    
    /////////////////////////////////////////////
    //z model
    /////////////////////////////////////////////
    
    ///////////////
    //update w 
    ///////////////

    //var = I + lambda' Psi^{-1} lambda
    for(i = 0; i < h; i++){
      for(j = 0; j < q; j++){
	tmp_hq[j*h+i] = lambda[j*h+i]/tauSq[i];
      }
    }
    
    dgemm_(&ytran, &ntran, &q, &q, &h, &one, tmp_hq, &h, lambda, &h, &zero, var, &q);

    //Add identity prior variance
    for(i = 0; i < q; i++){
      var[i*q+i] += 1.0;
    }

    //(var)^{-1}
    dpotrf_(&lower, &q, var, &q, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
    dpotri_(&lower, &q, var, &q, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}

    //Chol(var) for MV draw
    dcopy_(&qq, var, &inc, cholVar, &inc);
    dpotrf_(&lower, &q, cholVar, &q, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}  
    
    for(i = 0; i < n; i++){
      
      //mu
      for(j = 0; j < h; j++){
	tmp_h[j] = (z[i*h+j] - ddot_(&p, &X[np*j+i*p], &inc, &beta[j*p], &inc))/tauSq[j];
      }
      
      dgemv_(&ytran, &h, &q, &one, lambda, &h, tmp_h, &inc, &zero, mu, &inc);

      dsymv_(&lower, &q, &one, var, &q, mu, &inc, &zero, tmp_h, &inc);
      
      mvrnorm(&w[i*q], tmp_h, cholVar, q);
   
    }//end n

    ///////////////
    //update beta 
    ///////////////
    
    for(i = 0; i < h; i++){
      
      for(j = 0; j < n; j++){
    	tmp_n[j] = (z[j*h+i] - ddot_(&q, &w[j*q], &inc, &lambda[i], &h))/tauSq[i];
      }
      dgemv_(&ntran, &p, &n, &one, &X[np*i], &p, tmp_n, &inc, &zero, tmp_p, &inc); 
      
      for(j = 0; j < pp; j++){
    	tmp_pp[j] = XtX[i][j]/tauSq[i];
      }

      dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      dpotri_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
      
      dsymv_(&lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc);
      
      dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      
      mvrnorm(&beta[i*p], tmp_p2, tmp_pp, p);
    }
    
    ///////////////
    //update lambda 
    ///////////////

    zeros(tmp_qq, qq);
     
    for(k = 0; k < q; k++){
      for(l = 0; l < q; l++){
    	for(j = 0; j < n; j++){
    	  tmp_qq[k*q+l] += w[j*q+k]*w[j*q+l];
    	}
      }
    }
    
    for(i = 1; i < h; i++){

      zeros(tmp_q, q);
 
      if(i < q){//ll will be the dim of mu and var
    	ll = i;
      }else{
    	ll = q;
      }	
      
      //mu
      for(j = 0; j < n; j++){
    	tmp_n[j] = z[j*h+i] - ddot_(&p, &X[i*np+j*p], &inc, &beta[i*p], &inc);
	
    	if(i < q){
    	  tmp_n[j] -= w[j*q+i];
    	}
      }
      
      for(k = 0; k < ll; k++){
    	for(j = 0; j < n; j++){
    	  tmp_q[k] += w[j*q+k]*tmp_n[j];
    	}
      }

      for(k = 0; k < ll; k++){
    	tmp_q[k] /= tauSq[i];
      }

      //var      
      for(k = 0, l = 0; k < ll; k++){
    	for(j = 0; j < ll; j++, l++){
    	  tmp_qq2[l] = tmp_qq[k*q+j];
    	}
      }
      
      for(j = 0; j < ll*ll; j++){
    	tmp_qq2[j] /= tauSq[i];
      }
      
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

    /////////////////////
    //update tau^2
    /////////////////////
    
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
    }
    
    for(i = 0; i < nh; i++){
      tmp_nh[i] = z[i] - tmp_nh[i];
    }

    //Note, I'm just used a standard IG (because I'm lazy), but we should experiment with the half-t spelled out in the Taylor-Rodriguez appendix.
    for(i = 0; i < h; i++){
      tauSq[i] = 1.0/rgamma(tauSq_a+n/2.0, 1.0/(tauSq_b[i]+0.5*ddot_(&n, &tmp_nh[i], &h, &tmp_nh[i], &h)));
    }
    
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
    }

    if(s >= nSamplesStart){
      dcopy_(&P, beta, &inc, &betaSamples[sKeep*P], &inc);
      dcopy_(&hq, lambda, &inc, &lambdaSamples[sKeep*hq], &inc);
      dcopy_(&h, tauSq, &inc, &tauSqSamples[sKeep*h], &inc);
      dcopy_(&nq, w, &inc, &wSamples[sKeep*nq], &inc);
      dcopy_(&nh, tmp_nh, &inc, &fittedSamples[sKeep*nh], &inc);
      sKeep++;
    }
    
    /////////////////////////////////////////////
    //report
    /////////////////////////////////////////////
    
    if(status == nReport){
      cout << "percent complete: " << 100*s/nSamples << endl;
      cout << "---------------" << endl;
      status = 0;
    }
    status++;

  }
    
    writeRMatrix(outFile+"-beta", betaSamples, P, nSamplesKeep);
    writeRMatrix(outFile+"-tauSq", tauSqSamples, h, nSamplesKeep);
    writeRMatrix(outFile+"-w", wSamples, nq, nSamplesKeep);
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
