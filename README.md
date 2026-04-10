# MaxEntCppCompTest: Testing MaxEnt Equivalence

This repository is dedicated to testing the equivalence of essential operations between the two implementations of the Maximum Entropy Model:

- [@alrobles/maxentcpp](https://github.com/alrobles/maxentcpp) (C++ implementation)
- [@alrobles/Maxent](https://github.com/alrobles/Maxent) (JavaScript implementation)

## Objective

The objective of this repository is to ensure that the C++ implementation of MaxEnt (`@alrobles/maxentcpp`) performs equivalent operations as the JavaScript implementation (`@alrobles/Maxent`) by side-by-side comparisons in essential operations.

## Essentials of Comparison

The comparison will be carried out over the following key aspects and operations of the MaxEnt model:

### 1. Data Handling
- Parsing of input data
- Feature extraction processes

### 2. Training and Optimization
- Training algorithms (e.g., using gradient-based methods)
- Convergence of the optimization process
- Accuracy of calculated weights

### 3. Predictions
- Generating and scoring predictions
- Probabilistic distributions across categories

### 4. Performance Metrics
- Accuracy and precision of results
- Training and inference time benchmarks

## Structure and Contents

- **Test Benchmarks (`benchmarks/`)**: Contains scripts and test datasets to evaluate equivalency.
- **C++ Binding Tests (`cpp-tests/`)**: Comprehensive tests for `@alrobles/maxentcpp` functionality.
- **JavaScript Reference Tests (`js-tests/`)**: Reference tests for `@alrobles/Maxent` based on its implementation.

## How to Run the Tests

1. **Clone the Repository**

   ```bash
   git clone https://github.com/alrobles/maxentcppCompTest.git
   cd maxentcppCompTest
   ```

2. **Set Up the Environment**

   Install dependencies for both implementations:

   - For `@alrobles/maxentcpp`:
     ```
     # For example:
     cmake && make
     ```

   - For `@alrobles/Maxent`:
     ```
     npm install
     ```

3. **Run the Comparisons**

   Use the provided scripts in the `tests/side-by-side/` folder:

   ```bash
   ./run_comparison.sh
   ```

The results will be generated in the `outputs/` directory where differences, if any, will be highlighted.

## Development and Contributions

Contributions to further enhance the testing processes and to include other aspects of equivalency testing are welcome. Please ensure that your code follows the repository's guidelines.

---

For issues and queries, feel free to open an issue on this repository.
