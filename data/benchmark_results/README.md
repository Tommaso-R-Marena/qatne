# Benchmark Results

This directory contains benchmark results for QATNE algorithm.

## Structure

- `h2_results.json` - Hydrogen molecule results
- `lih_results.json` - Lithium hydride results
- `comparison_data.csv` - Comparison with classical methods

## Running Benchmarks

To generate your own benchmark results:

```bash
python scripts/run_benchmarks.py --molecule H2 --output data/benchmark_results/
```

## Data Format

Results are stored in JSON format:

```json
{
  "molecule": "H2",
  "bond_length": 0.735,
  "ground_energy": -1.137283834,
  "exact_energy": -1.137283834,
  "error": 1.2e-06,
  "iterations": 156,
  "time_seconds": 45.3,
  "bond_dimension_final": 16
}
```
