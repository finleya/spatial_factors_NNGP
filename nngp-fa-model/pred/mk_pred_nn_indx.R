library(RANN)

## n is obs, n.0 is prediction locations.

coords.0 <- read.table("../sim_data/coords.ho")
coords <- read.table("../sim_data/coords.obs")
n <- nrow(coords)
m <- 15 ## Number of neighbors.

nn.indx <- nn2(coords, coords.0, k = m)$nn.idx

## Adjust to zero index for c code.
write.table(nn.indx-1, "nn.indx", row.names=F, col.names=F, sep="\t")
