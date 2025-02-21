/**********************************
 * Description: An out-of-place implementation of QFT addition.
 * Author: Jacob Collins
 **********************************/
#include <cudaq.h>
#include <cudaq/algorithms/draw.h>

#include <math.h>
#include <numbers>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// # define M_PIl          3.141592653589793238462643383279502884L /* pi */

#define ENABLE_DEBUG true
#define NUMBER_OF_SHOTS 3

/**************************************************
******************* HELPER FUNCS ******************
***************************************************/

// Convert some value to a string
template <typename T>
std::string to_string(T value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// Convert nanoseconds to a string displaying subdivisions of time
std::string format_time(long long nanoseconds, bool sep=false) {
  long long milliseconds = nanoseconds / 1'000'000;  // From ns to ms
  long long seconds = milliseconds / 1'000;          // From ms to seconds
  milliseconds = milliseconds % 1'000;               // Remaining milliseconds

  long long remaining_nanoseconds =
      nanoseconds % 1'000'000;  // Remaining nanoseconds
  long long microseconds = remaining_nanoseconds / 1'000;  // From ns to µs
  remaining_nanoseconds = remaining_nanoseconds % 1'000;   // Remaining ns

  // Create the formatted string
  if (sep) {
    return to_string(seconds) + "s " + to_string(milliseconds) + "ms " +
            to_string(microseconds) + "µs " + to_string(remaining_nanoseconds) +
            "ns";
  } else {
    float s = (float) nanoseconds / 1'000'000'000;
    return to_string(s) + "s";
  }
}

// Convert an unordered_map to a sorted vector of tuples.
// ( The unordered map is the result of cudaq::sample() )
std::vector<std::tuple<std::string, size_t>> sort_map(const std::unordered_map<std::string, size_t>& myMap) {
    // Create a vector of tuples from the unordered_map
    std::vector<std::tuple<std::string, size_t>> vec;
    for (const auto& pair : myMap) {
        vec.emplace_back(pair.first, pair.second);
    }

    // Sort the vector in descending order based on the size_t value
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b); // Compare the size_t values
    });

    return vec;
}

// Convert value to binary string
template <typename T>
std::string bin_str(T val, int nbits, bool qorder=true) {
  std::stringstream ss;
  if (qorder) {
    for (int i = nbits; i >= 1; --i) {
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
int bin_to_int(std::string &s, bool qorder=true) {
  int result = 0;
  int len = s.length();
  if (qorder) {
    for (int i = len-1; i >= 0; --i) {
      if (s[i] == '1') {
        result += pow(2, i);
      }
    }
    return result;
  }
  for (int i = 0; i < len; ++i) {
    if (s[i] == '1') {
      result += pow(2, i);
    }
  }
  return result;
}


/****************** CUDAQ FUNCS ******************/
// Apply NOT-gates in accordance with bit-pattern of given integer.
// Bit-order is reversed. i.e. 3 (00011) would be (11000)
__qpu__ void setInt(const long val, cudaq::qview<> qs, bool qorder=true) {
  if (qorder) {
    // Iterate through bits in val
    for (int i = qs.size(); i >= 1; --i) {
      // Bit-shift for single bitwise AND to apply X on correct qubits
      auto target_bit_set = (1 << (qs.size() - i)) & val;
      if (target_bit_set) x(qs[qs.size() - i]);
    }
  } else {
    // Iterate through bits in val
    for (int i = 1; i <= qs.size(); ++i) {
      // Bit-shift for single bitwise AND to apply X on correct qubits
      auto target_bit_set = (1 << (qs.size() - i)) & val;
      if (target_bit_set) x(qs[qs.size() - i]);
    }
  }
}

// Converted from qml function
// https://docs.pennylane.ai/en/stable/_modules/pennylane/templates/subroutines/qft.html#QFT
__qpu__ void quantumFourierTransform(cudaq::qview<> qs) {
  const int nbits = qs.size();
  int shift_len = nbits-1;
  double shifts[nbits-1];
  double idx;
  // Calculate phase shifts 
  for (int i = 0; i < shift_len; ++i) {
    idx = i+2;
    shifts[i] = (double) 2 * std::numbers::pi * pow(2, -idx); 
    if (ENABLE_DEBUG) { printf("QFT PHASE %d: %lf\n", i, shifts[i]); }
  }
  // Apply phase shifts
  for (int i = 0; i < nbits; ++i) {
    h(qs[i]);
    for (int j = i + 1; j < nbits; ++j) {
      cr1(shifts[j-(i+1)], qs[j], qs[i]);
    }
  }
  // SWAP first half and reversed last half
  for (int i = 0; i < nbits / 2; ++i) {
    swap(qs[i], qs[(nbits-1)-i]);
  }
}

// Based on pennylane.ai implementation
// https://pennylane.ai/qml/demos/tutorial_qft_arithmetics
// ~WORKS PERFECTLY~
struct QFTAdder {
  void operator()(cudaq::qview<> x_reg, cudaq::qview<> y_reg, cudaq::qview<> z_reg) __qpu__ {
    const int nbits_y = y_reg.size();
    const int nbits_z = z_reg.size();
    double lambda;
    for (int j = 0; j < nbits_y; ++j) {
      for (int k = 0; k < nbits_y-j; ++k) {
        lambda = std::numbers::pi / pow(2, k);
        rz<cudaq::ctrl>(lambda, y_reg[j], z_reg[j+k]);
      }
    }
    for (int j = 0; j < nbits_y; ++j) {
      lambda = std::numbers::pi / pow(2, j);
      rz<cudaq::ctrl>(lambda, y_reg[nbits_y - (j - 1)], z_reg[nbits_y]);
    }
  }
};

/****************** CUDAQ STRUCTS ******************/
// Driver for adder
struct runAdder {
  __qpu__ auto operator()(const long x, const long y,
                          const int nbits_x, const int nbits_y, const int nbits_z) {
    // 1. Initialize Registers
    QFTAdder add_op;
    cudaq::qvector q_reg(nbits_x + nbits_y + nbits_z);  // Value 1 reg
    cudaq::qview x_reg = q_reg.front(nbits_x);
    cudaq::qview y_reg = q_reg.slice(nbits_x, nbits_y);
    cudaq::qview z_reg = q_reg.back(nbits_z);  // Value 2 reg
    setInt(x, x_reg);
    setInt(y, y_reg);
    // setInt(4, z_reg);

    // 2. QFT
    // for (int i = 0; i < 10; ++i) {
    quantumFourierTransform(z_reg);

    // 2. Add
    // add_op(x_reg, y_reg, z_reg);
    // cudaq::adjoint(add_op, y_reg, z_reg, c);

    // 4. IQFT
    // cudaq::adjoint(quantumFourierTransform, z_reg);
    // }
    // 3. Measure
    // mz(q_reg);
  }
};

void display_full_results(std::vector<std::tuple<std::string, size_t>> results, long x, long y, int nbits_x, int nbits_y, int nbits_z, size_t n_printed=5) {
  size_t n_shots = NUMBER_OF_SHOTS;
  size_t total_correct = 0;
  int i = 0;
  for (auto item : results) {
    // Binary result string
    std::string result = std::get<0>(item);
    // Count of this outcome being measured
    size_t count = std::get<1>(item);
    // Parse
    std::string x_out = result.substr(0, nbits_x);
    std::string y_out = result.substr(nbits_x, nbits_y);
    std::string z_out = result.substr(nbits_y, nbits_z);
    int x_val = bin_to_int(x_out);
    int y_val = bin_to_int(y_out);
    int z_val = bin_to_int(z_out);
    // % of whole
    if (i < n_printed) {
      printf("%d + %d = %d (%lu/%lu = %.2f%%)\n", x_val, y_val, z_val, count, n_shots, (float) 100 * count / n_shots);
      if (ENABLE_DEBUG) {
        printf("  Full result: %s\n", result.c_str());
        printf("  (R1) x: %d (%s)\n", x_val, bin_str(x_val, nbits_x).c_str());
        printf("  (R2) y: %d (%s)\n", y_val, bin_str(y_val, nbits_y).c_str());
        printf("  (R3) z: %d (%s)\n", z_val, bin_str(z_val, nbits_z).c_str());
      }
    }
    if (z_val == x+y) {
      total_correct += count;
    }
    i++;
  }
  if (n_printed < results.size()) {
    printf("More results hidden...\n");
  }
  // The percentage of results that were correct.
  printf("%lu / %lu Correct. (%.2f%%)\n", total_correct, n_shots, (float) 100 * total_correct / n_shots);
}

void run_QFT_adder(long x, long y) {
  // Necessary # bits computed based on input values. Min 1.
  int nbits_x = ceil(log2(max(std::vector<long>({x, y, 1})) + 1));
  int nbits_y = nbits_x;
  int nbits_z = nbits_x + 1;
  long z = x + y;

  printf("\nVERIFIED INPUTS\n");
  printf("x: %ld (%s), nbits=%d\n", x, bin_str(x, nbits_x).c_str(), nbits_x);
  printf("y: %ld (%s), nbits=%d\n", y, bin_str(y, nbits_y).c_str(), nbits_y);
  printf("\nEXPECTED VALUES\n");
  printf("z: %ld (%s)\n", z, bin_str(z, nbits_z).c_str());
  printf("Adding values: %ld + %ld = %ld\n", x, y, z);
  printf("Expected Full Out: (%s_%s_%s)\n", bin_str(x, nbits_x).c_str(),
         bin_str(y, nbits_y).c_str(), bin_str(z, nbits_z).c_str());

  // Draw circuit and view statevector
  std::cout << cudaq::draw(runAdder{}, x, y, nbits_x, nbits_y, nbits_z);
  auto state = cudaq::get_state(runAdder{}, x, y, nbits_x, nbits_y, nbits_z);
  state.dump();
  
  // GENERATE AND RUN CIRCUIT
  auto start = std::chrono::high_resolution_clock::now();

  int n_shots = NUMBER_OF_SHOTS; // Get a lot of samples
  auto counts = cudaq::sample(n_shots, runAdder{}, x, y, nbits_x, nbits_y, nbits_z);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nAdder finished in %s.\n", format_time(duration).c_str());
  std::vector<std::tuple<std::string, size_t>> results = sort_map(counts.to_map());
  printf("\nMEASURED RESULTS\n");
  display_full_results(results, x, y, nbits_x, nbits_y, nbits_z, 10);
  printf("\n");
  // // REVIEW RESULTS
  // std::string result = counts.most_probable();
  // printf("Full out: (%s)\n", result.c_str());
  // std::string val2_out = result.substr(0, nbits_y);
  // std::string sum_out = result.substr(nbits_y, nbits_z);
  // printf("Sum: %d (%s)\n", bin_to_int(sum_out), sum_out.c_str());
}

/****************** CUDAQ STRUCTS ******************/
int main(int argc, char *argv[]) {
  // PARSE INPUT VALUES
  // Default search value
  printf("Usage: ./inverse_add.x [x] [y]\n");
  long x=1, y=2;
  if (argc >= 3) {
    x = strtol(argv[1], nullptr, 10);
    y = strtol(argv[2], nullptr, 10);
  }
  printf("This will attempt to use QFT to compute x + y = z (Out-of-place).\n");
  run_QFT_adder(x, y);
}
