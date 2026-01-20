# Car Insurance Data

This directory should contain the car insurance CSV files for conditional independence testing.

## Expected Files

The data files should be named `{state}-per-zip.csv` where `{state}` is a two-letter state code.

Expected files:
- `ca-per-zip.csv` - California
- `il-per-zip.csv` - Illinois
- `mo-per-zip.csv` - Missouri
- `tx-per-zip.csv` - Texas

## Data Source

The data comes from: https://github.com/felipemaiapolo/cit/tree/main (MIT license)

Original repository: https://github.com/romanpogodin/kernel-ci-testing

## Required Columns

Each CSV file should contain the following columns:
- `state_risk`: Risk score for the state/region (conditioning variable Z)
- `combined_premium`: Insurance premium amount (outcome variable Y)
- `minority`: Binary indicator for minority status (treatment variable X)
- `companies_name`: Name of the insurance company

## Testing Scenario

This dataset tests whether insurance premiums (Y) are conditionally independent of 
minority status (X) given state risk (Z). This is a fairness testing scenario.

- **Type 1 (H0)**: Simulated conditional independence by shuffling Y within Z bins
- **Type 2 (H1)**: Real data testing actual conditional independence
