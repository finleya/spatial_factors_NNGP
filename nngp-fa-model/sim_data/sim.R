rm(list=ls())
library(spBayes)

rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

set.seed(1)

##number locations
n <- 2000
coords <- cbind(runif(n,0,1), runif(n,0,1))
coords <- coords[order(coords[,1]),]

D <- as.matrix(dist(coords))

##number of outcomes at each location
h <- 10

##X
##assuming p = 3, i.e., and intercept and two covariates
x <- cbind(1, rnorm(n), rnorm(n))
V <- vector("list", h)

for(i in 1:h){
    V[[i]] <- x
}

X <- mkMvX(V)

##beta, tau^2
B <- as.vector(t(cbind(rnorm(h), rnorm(h), rnorm(h))))
tau.sq <- seq(0.1,1,length.out=h)

##spatial stuff assume q columns in lambda
q <- 4
phi <- 3/seq(0.5, 1, length.out=q)

##w
w <- rep(0, q*n)
for(i in 1:q){
    R <- exp(-phi[i]*D)
    w[seq(i, q*n, by=q)] <- rmvn(1, rep(0,n), R)
}

##Lambda
lambda <- matrix(rnorm(h*q, 0, 0.5), h, q)
for(i in 1:q){
    for(j in 1:q){
        if(j == i){
            lambda[i, j] <- 1
        }

        if(i < j){
            lambda[i, j] <- 0
        }
    }
}

v <- rep(0, h*n)
for(i in 1:n){
    v[((i-1)*h+1):(i*h)] <- lambda%*%w[((i-1)*q+1):(i*q)]
}

z <- rnorm(n*h, X%*%B + v, sqrt(tau.sq))

##write stuff out
set.seed(1)
ho <- sample.int(n,floor(0.5*n))

z.m <- matrix(z, nrow = h)
w.m <- matrix(w, nrow = q)

z.obs <- as.numeric(z.m[,-ho])
w.obs <- as.numeric(w.m[,-ho])
coords.obs <- coords[-ho,]
V.obs <- lapply(V, function(x){x[-ho,]})

z.ho <- as.numeric(z.m[,ho])
w.ho <- as.numeric(w.m[,ho])
coords.ho <- coords[ho,]
V.ho <- lapply(V, function(x){x[ho,]})

a <- x[ho,]
b <- do.call("rbind", V.ho)

write.table(z.obs, "z.obs", row.names=F, col.names=F, sep="\t") ## Stacked by location.
write.table(coords.obs, "coords.obs", row.names=F, col.names=F, sep="\t") 
write.table(t(do.call("rbind", V.obs)), "X.obs", row.names=F, col.names=F, sep="\t") ## Stacked by location!
write.table(w.obs, "w.obs", row.names=F, col.names=F, sep="\t") ## Stacked by location
write.table(rep(0,length(w.obs)), "w.starting", row.names=F, col.names=F, sep="\t")

write.table(z.ho, "z.ho", row.names=F, col.names=F, sep="\t") ## Stacked by location.
write.table(coords.ho, "coords.ho", row.names=F, col.names=F, sep="\t")
write.table(t(do.call("rbind", V.ho)), "X.ho", row.names=F, col.names=F, sep="\t") ## Stacked by location!
write.table(w.ho, "w.ho", row.names=F, col.names=F, sep="\t") ## Stacked by location

write.table(lambda, "lambda", row.names=F, col.names=F, sep="\t")

lambda.starting <- matrix(rnorm(h*q), h, q)
for(i in 1:q){
    for(j in 1:q){
        if(j == i){
            lambda.starting[i, j] <- 1
        }

        if(i < j){
            lambda.starting[i, j] <- 0
        }
    }
}

write.table(lambda.starting, "lambda.starting", row.names=F, col.names=F, sep="\t")
write.table(tau.sq, "tauSq", row.names=F, col.names=F, sep="\t")
write.table(rep(mean(tau.sq), length(tau.sq)), "tauSq.starting", row.names=F, col.names=F, sep="\t")
write.table(tau.sq, "tauSq.b", row.names=F, col.names=F, sep="\t")
write.table(phi, "phi", row.names=F, col.names=F, sep="\t")
write.table(rep(mean(phi), length(phi)), "phi.starting", row.names=F, col.names=F, sep="\t")
write.table(B, "beta", row.names=F, col.names=F, sep="\t")
write.table(rep(0, length(B)), "beta.starting", row.names=F, col.names=F, sep="\t")


