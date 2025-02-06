CXX = g++
OPTFLAGS = -O3 -fopenmp
CXXFLAGS = -I/opt/OpenBLAS/include 
LIBS = -I/opt/OpenBLAS/include -I/usr/local/lib/R/include -L /opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran -L/usr/local/lib -lRmath -L/usr/lib -lR 

nngp-fa: nngp-fa.o kvpar.o 
	$(CXX) $(OPTFLAGS) -o nngp-fa nngp-fa.o kvpar.o $(LIBS) 

nngp-fa.o: nngp-fa.cpp
	$(CXX) $(OPTFLAGS) nngp-fa.cpp -c $(LIBS)

kvpar.o: ../libs/kvpar.cpp
	$(CXX) $(OPTFLAGS) ../libs/kvpar.cpp -c

clean:
	rm -rf *o nngp-fa
