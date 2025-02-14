# Makefile for CudaQ
CFLAGS = -Wall -g
CXXFLAGS = -Wall
LDLIBS = 
CC = nvcc
CXX = nvq++
# NVQFLAGS = --target nvidia
NVQFLAGS = 

default: QFT_factor

# PRODUCT = QFT_addition QFT_factor QFT_multiplication sp_factorization test-install inverse_add basic_addition groveer shors QFT_scaled_addition

# make <filename>.o
%.x: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(NVQFLAGS)
    
# make <filename> == make <filename>.x
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@.x $(NVQFLAGS)

.PHONY: default clean

clean:
	rm -f *.o *.tmp *.x
