# Makefile for CudaQ
CFLAGS     = -Wall -g
CXXFLAGS   = -Wall
LDLIBS     =
CC         = nvcc
CXX        = nvq++
NVQFLAGS   = --target nvidia
# NVQFLAGS = 

# PRODUCT = test-install inverse_add basic_addition QFT_addition QFT_factor QFT_multiplication sp_factorization grover QFT_scaled_addition
PRODUCT = test-install inverse_add basic_addition QFT_addition QFT_exploration

.PHONY: all clean

# List C++ source files (assumed to have the .cpp extension).
CPPFILES := $(wildcard *.cpp)

# Default rule: build all products (.x executables)
all: $(PRODUCT:%=%.x)
# all: $(CPPFILES:%=%.x)

# Rule: compile non-header (.cpp) files directly to executables,
# linking with the header object files
%.x: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(NVQFLAGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@.x $(NVQFLAGS)

# Clean up generated files
clean:
	rm -f *.o *.x
