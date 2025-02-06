CXX = g++
OPTFLAGS = -O3 -fopenmp
CXXFLAGS = -I/opt/OpenBLAS/include 
LIBS = -I/opt/OpenBLAS/include -I/usr/local/lib/R/include -L /opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran -L/usr/local/lib -lRmath -L/usr/lib -lR  

fa: fa.o kvpar.o
	$(CXX) $(OPTFLAGS) -o fa fa.o kvpar.o $(LIBS) 

fa.o: fa.cpp
	$(CXX) $(OPTFLAGS) fa.cpp -c $(LIBS)

kvpar.o: ../libs/kvpar.cpp
	$(CXX) $(OPTFLAGS) ../libs/kvpar.cpp -c

clean:
	rm -rf *o fa
