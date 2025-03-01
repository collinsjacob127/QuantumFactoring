#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>
#include <stdio.h>

__qpu__ void set_int(const long val, cudaq::qview<> qs, bool qorder=true) {
  if (qorder) {
    // Iterate through bits in val
    for (int i = 1; i <= qs.size(); ++i) {
      // Bit-shift for single bitwise AND to apply X on correct qubits
      auto target_bit_set = (1 << (qs.size() - i)) & val;
      if (target_bit_set) x(qs[i-1]);
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

__qpu__ void kernel(int qubit_count, long val) {
  cudaq::qvector qubits(qubit_count);
  set_int(val, qubits);
  mz(qubits);
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

int min_bits(long x) {
    return ceil(log2(max(std::vector<long>({x, 1})) + 1));
}

int main(int argc, char *argv[]) {
  long val = 1 < argc ? atol(argv[1]) : 9;
  int nbits = min_bits(val);

  printf("Setting %ld with %d bits.\n", val, nbits);

//   auto result = cudaq::sample(kernel, min_bits(val), val);
//   result.dump(); // Example: { 11:500 00:500 }
  std::cout << cudaq::draw(kernel, min_bits(val), val);
}
