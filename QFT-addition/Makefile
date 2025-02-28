# Makefile for CudaQ
CFLAGS     = -Wall -g
CXXFLAGS   = -Wall
LDLIBS     =
CC         = nvcc
CXX        = nvq++
NVQFLAGS   = --target nvidia
# NVQFLAGS = 

# List C++ source files (assumed to have the .cpp extension).
CPPFILES := $(wildcard *.cpp)

.PHONY: all clean


# Default rule: build all products (.x executables)
#all: $(PRODUCT:%=%.x)
all: $(CPPFILES:%.cpp=%.x)

# Rule: compile non-header (.cpp) files directly to executables,
# linking with the header object files
%.x: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(NVQFLAGS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@.x $(NVQFLAGS)

# Clean up generated files
clean:
	rm -f *.o *.x
