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
n <- 1000
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
B <- as.vector(t(cbind(rnorm(h, 0, 1), rnorm(h, 0, 1), rnorm(h, 0, 1))))

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
lambda <- matrix(rnorm(h*q, 0, 1), h, q)
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

prob <- 1/(1+exp(-(X%*%B + v)))

z <- rbinom(n*h, size=rep(1,n*h), prob=prob)

##write stuff out
write.table(v, "Zw", row.names=F, col.names=F, sep="\t")
write.table(z, "z", row.names=F, col.names=F, sep="\t")
write.table(rep(0,length(z)), "missing", row.names=F, col.names=F, sep="\t")
write.table(coords, "coords", row.names=F, col.names=F, sep="\t")
write.table(t(do.call("rbind", V)), "X", row.names=F, col.names=F, sep="\t")

write.table(w, "w", row.names=F, col.names=F, sep="\t")
write.table(rep(0,length(w)), "w.starting", row.names=F, col.names=F, sep="\t")

write.table(lambda, "lambda", row.names=F, col.names=F, sep="\t")

lambda.starting <- matrix(0, h, q)
diag(lambda.starting) <- 1
## for(i in 1:q){
##     for(j in 1:q){
##         if(j == i){
##             lambda.starting[i, j] <- 1
##         }

##         if(i < j){
##             lambda.starting[i, j] <- 0
##         }
##     }
## }
write.table(lambda.starting, "lambda.starting", row.names=F, col.names=F, sep="\t")

write.table(phi, "phi", row.names=F, col.names=F, sep="\t")
write.table(rep(mean(phi), length(phi)), "phi.starting", row.names=F, col.names=F, sep="\t")
            
write.table(B, "beta", row.names=F, col.names=F, sep="\t")
write.table(rep(0, length(B)), "beta.starting", row.names=F, col.names=F, sep="\t")


