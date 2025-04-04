// Compile and run with:
// ```
// nvq++ grover.cpp -o grover.x && ./grover.x
// ```

// Base includes
#include <cmath>
#include <cudaq.h>
#include <numbers>
// Custom includes
#include <vector>
#include <numeric>
#include <sstream>
#include <string>
#include <iostream>
#include <bitset>

// Convert value to binary string
template <typename T>
std::string bin_str(T val, int nbits) {
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

__qpu__ void reflect_about_uniform(cudaq::qvector<> &qs) {
  auto ctrlQubits = qs.front(qs.size() - 1);
  auto &lastQubit = qs.back();

  // Compute (U) Action (V) produces
  // U V U::Adjoint
  cudaq::compute_action(
      [&]() {
        h(qs);
        x(qs);
      },
      [&]() { z<cudaq::ctrl>(ctrlQubits, lastQubit); });
}

struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, CallableKernel &&oracle) {
    int n_iterations = round(0.25 * std::numbers::pi * sqrt(2 ^ n_qubits));

    cudaq::qvector qs(n_qubits);
    h(qs);
    for (int i = 0; i < n_iterations; i++) {
      oracle(qs);
      reflect_about_uniform(qs);
    }
    mz(qs);
  }
};

struct oracle {
  const long target_state;
  const std::vector<long> arr;

  void operator()(cudaq::qvector<> &qs) __qpu__ {
    cudaq::compute_action(
        // Define good search state (secret)
        [&]() {
          for (int i = 1; i <= qs.size(); ++i) {
            auto target_bit_set = (1 << (qs.size() - i)) & target_state;
            if (!target_bit_set)
              x(qs[i - 1]);
          }
        },
        // Controlled z, sends search result to tgt bit
        [&]() {
          auto ctrlQubits = qs.front(qs.size() - 1);
          z<cudaq::ctrl>(ctrlQubits, qs.back());
        });
  }
};

long max(std::vector<long> arr) {
  long max = arr[0];
  for (auto &v : arr) {
    if (v > max) { max = v; }
  }
  return max;
}

template <typename T>
std::string arrayToString(std::vector<T> arr, bool binary, int nbits) {
  std::stringstream ss;
  if (binary) {
    for (int i = 0; i < arr.size(); i++) {
      ss << std::bitset<sizeof(long) * 8>(arr[i]).to_string().substr(sizeof(T)*8-nbits, nbits);
      if (i < arr.size()-1) {
        ss << ", ";;
      }
    }
  } else {
    ss << arr[0];
    for (int i = 1; i < arr.size(); i++) {
      ss << ", " << arr[i];
    }
  }
  return ss.str();
}

int main(int argc, char *argv[]) {
  // Set up the list of values to search through
  std::vector<long> search_vals = {7, 4, 2, 9, 10, 3, 15, 2, 5};
  std::vector<long> index_vals(search_vals.size());
  std::iota(index_vals.begin(), index_vals.end(), 0);
  long secret;

  // Set up value to search for, secret defaults to 3
  printf("Usage: ./grover.x [search value]\n\n");
  if (argc >= 2) {
    secret = strtol(argv[1], NULL, 10);
  } else {
    secret = 9;
  }
  int nbits_val = ceil(log2(max(std::vector<long>({max(search_vals), secret})) + 1));
  int nbits_index = ceil(log2(search_vals.size()));
  int nbits = ceil(log2(secret+1));

  // Helpful output
  std::cout << "Search vals: " << arrayToString(search_vals, false, nbits_val) << std::endl;
  std::cout << "Index vals: " << arrayToString(index_vals, false, nbits_index) << std::endl;
  std::cout << "Search vals: " << arrayToString(search_vals, true, nbits_val) << std::endl;
  std::cout << "Index vals: " << arrayToString(index_vals, true, nbits_index) << std::endl;
  printf("Secret: %ld (%s)\n", secret, bin_str(secret, nbits).c_str());
  printf("Nbits: %d\n", nbits);
  
  // Generate Circuits and run
  oracle compute_oracle{.target_state = secret, .arr = search_vals};
  auto counts = cudaq::sample(run_grover{}, nbits, compute_oracle);
  std::string result = counts.most_probable();
  printf("Found string %s (%ld)\n", result.c_str(), strtol(result.c_str(), NULL, 2));
}
