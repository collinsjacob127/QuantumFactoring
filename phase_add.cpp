/**********************************
 * Description: Example Adder using CudaQ
 * Author: Jacob Collins
 * Instructions:
 *   Compile and run with:
 *   ```
 *   $> make phase_add
 *   $> ./phase_add # Uses default values
 *   OR
 *   $> ./phase_add 00101 11101 # Takes binary input
 *   ```
 **********************************/
#include <cudaq.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
// # define M_PIl          3.141592653589793238462643383279502884L /* pi */

#define DEBUG true

/****************** HELPER FUNCS ******************/
// Convert some value to a string
template <typename T>
std::string toString(T value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// Convert nanoseconds to a string displaying subdivisions of time
std::string formatTime(long long nanoseconds) {
    
    long long milliseconds = nanoseconds / 1'000'000;  // From ns to ms
    long long seconds = milliseconds / 1'000;          // From ms to seconds
    milliseconds = milliseconds % 1'000;               // Remaining milliseconds

    long long remaining_nanoseconds = nanoseconds % 1'000'000; // Remaining nanoseconds
    long long microseconds = remaining_nanoseconds / 1'000;    // From ns to µs
    remaining_nanoseconds = remaining_nanoseconds % 1'000;     // Remaining ns

    // Create the formatted string
    return toString(seconds) + "s " +
           toString(milliseconds) + "ms " +
           toString(microseconds) + "µs " +
           toString(remaining_nanoseconds) + "ns";
}

// Convert value to binary string
template <typename T>
std::string binStr(T val, int nbits) {
  std::stringstream ss;
  for (int i = 1; i <= nbits; ++i) {
    // Shift through the bits in val
    auto target_bit_set = (1 << (nbits - i)) & val;
    // Add matching val to string
    if (target_bit_set) {
      ss << '1';
    } else {
      ss << '0';
    }
  }
  return ss.str();
}

// Return max value in an array
template <typename T>
T max(std::vector<T> arr) {
  T max = arr[0];
  for (auto &v : arr) {
    if (v > max) {
      max = v;
    }
  }
  return max;
}

// Convert bin string to int. 1101 -> 13
int binToInt(std::string &s) {
  int result = 0;
  int len = s.length();
  for (int i = 0; i < len; ++i) {
    if (s[i] == '1') {
      result += pow(2, len - 1 - i);
    }
  }
  return result;
}

/****************** CUDAQ FUNCS ******************/

// Apply NOT-gates in accordance with bit-pattern of given integer.
__qpu__ void setInt(const long val, cudaq::qview<> qs) {
  // Iterate through bits in val
  for (int i = 1; i <= qs.size(); ++i) {
    // Bit-shift for single bitwise AND to apply X on correct qubits
    auto target_bit_set = (1 << (qs.size() - i)) & val;
    // Apply X if bit i is valid
    if (target_bit_set) {
      x(qs[i - 1]);
    } 
  }
}

struct Carry {
  const int nbits_v;

  void operator()(cudaq::qview<> v_reg1, cudaq::qview<> v_reg2,
                 cudaq::qview<> c_reg, int i) __qpu__ {
    x<cudaq::ctrl>(v_reg1[i], v_reg2[i], c_reg[i]);
    x<cudaq::ctrl>(v_reg1[i], v_reg2[i]);
    x<cudaq::ctrl>(c_reg[i + 1], v_reg2[i], c_reg[i]);
  }
};

// Bitwise addition of v_reg1 and v_reg2.
// Output is {c_reg[0], v_reg2}
struct Adder {

  void operator()(cudaq::qview<> v_reg1, cudaq::qview<> v_reg2,
                  cudaq::qview<> c_reg) __qpu__ {
    const int nbits_v = v_reg1.size();
    Carry carry_op{.nbits_v = nbits_v};
    // Store all the carries in c_reg
    for (int i = nbits_v - 1; i >= 0; --i) {
      carry_op(v_reg1, v_reg2, c_reg, i);
      // x<cudaq::ctrl>(v_reg1[i], v_reg2[i], c_reg[i]);
      // x<cudaq::ctrl>(v_reg1[i], v_reg2[i]);
      // x<cudaq::ctrl>(c_reg[i + 1], v_reg2[i], c_reg[i]);
    }
    // Update reg 2 highest-order bit
    x<cudaq::ctrl>(v_reg1[0], v_reg2[0]);
    for (int i = 0; i < nbits_v; ++i) {
      // Perform sum with carries, send to reg 2
      x<cudaq::ctrl>(v_reg1[i], v_reg2[i]);
      x<cudaq::ctrl>(c_reg[i + 1], v_reg2[i]);
      if (i < nbits_v - 1) {
        // Undo carries, except highest-order carry bit
        cudaq::adjoint(carry_op, v_reg1, v_reg2, c_reg, i+1);
      }
    }
    return;
  }
};

struct ScaledAdder {

  void operator()(cudaq::qview<> y_reg, cudaq::qview<> z_reg,
                  long c) __qpu__ {
    const int nbits_y = y_reg.size();
    const int nbits_z = z_reg.size();
    int y_diff, z_diff;
    int j;
    double phase;
    
    for (int y_ind = nbits_y-1; y_ind >= 0; --y_ind) {
      for (int z_ind = nbits_z-1; z_ind >= y_ind; --z_ind) {
        z_diff = nbits_z - z_ind;
        j = -(z_ind - y_ind - (nbits_z - nbits_y));
        // phase = (double) c * M_PI / (double) pow(2, j); 
        phase = c * M_PI / pow(2, j); 
        if (DEBUG) { printf("Phase: %lf\n  y_i: %d\n  z_i: %d\n  j: %d\n", phase, y_ind, z_ind, j); }
        r1<cudaq::ctrl>(phase, y_reg[y_ind], z_reg[z_ind]);
      }
    }
    // for (int y_ind = 0; y_ind < nbits_y; ++y_ind) {
    //   for (int z_ind = 0; z_ind < nbits_z; ++z_ind) {
    //     j = z_ind - y_ind;
    //     phase = c * M_PI / pow(2, j); 
    //     if (DEBUG) { printf("Phase: %lf\n  y_i: %d\n  z_i: %d\n  j: %d\n", phase, y_ind, z_ind, j); }
    //     r1<cudaq::ctrl>(phase, y_reg[y_ind], z_reg[z_ind]);
    //   }
    // }
  }
};

/****************** CUDAQ STRUCTS ******************/
// Driver for adder
struct run_adder {
  __qpu__ auto operator()(const long y, const long z, const long c, const int nbits_y, const int nbits_z) {
    // 1. Initialize Registers
    ScaledAdder add_op;
    cudaq::qvector y_reg(nbits_y);  // Value 1 reg
    cudaq::qvector z_reg(nbits_z);  // Value 2 reg
    setInt(y, y_reg);
    setInt(z, z_reg);

    // 2. Add
    add_op(y_reg, z_reg, c);
    // 3. Measure
    mz(y_reg, z_reg);
  }
};

void runScaledAdder(long y, long z, long c) {
  // PARSE INPUT VALUES
  // Values to add, optionally passed in cmdline
  // long y = 0b1000;
  // long z = 0b0100;
  // long c = 0b0010;

  // Necessary # bits computed based on input values. Min 1.
  int nbits_y = ceil(log2(max(std::vector<long>({y, z, 1})) + 1));
  int nbits_z = 2*nbits_y+3;

  printf("y: %ld (%s)\n", y, binStr(y, nbits_y).c_str());
  printf("z: %ld (%s)\n", z, binStr(z, nbits_z).c_str());
  printf("z+(c*y): %ld (%s)\n", z+y*c, binStr(z+y*c, nbits_z).c_str());
  printf("Adding values: %ld + %ld x %ld = %ld\n", z, y, c, z+c*y);
  printf("Expected Full Out: (%s%s)\n", binStr(y, nbits_y).c_str(), binStr(z+y*c, nbits_z).c_str());

  // GENERATE AND RUN CIRCUIT
  auto start = std::chrono::high_resolution_clock::now();

  auto counts = cudaq::sample(run_adder{}, y, z, c, nbits_y, nbits_z);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Adder finished in %s.\n", formatTime(duration).c_str());
  
  // REVIEW RESULTS
  std::string result = counts.most_probable();
  printf("Full out: (%s)\n", result.c_str());
  std::string val2_out = result.substr(0, nbits_y);
  std::string sum_out = result.substr(nbits_y, nbits_z);
  printf("Sum: %d (%s)\n", binToInt(sum_out), sum_out.c_str());
}

/****************** CUDAQ STRUCTS ******************/
int main() {
  // 4 + 3 * 5 = 19
  runScaledAdder(3, 4, 5);
  printf("\n");
  // 2 + 1*3 = 5
  runScaledAdder(1, 2, 3);
  printf("\n");
  // 3 + 2*4
  runScaledAdder(2, 3, 4);
  printf("\n");
}
