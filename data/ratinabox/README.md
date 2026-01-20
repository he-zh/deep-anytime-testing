# RatInABox Data

This directory should contain pre-generated neural simulation data files from RatInABox.

## Expected File Format

Files should be named: `{n_cells}_cells_{max_n_points}_points_noise_{noise_std}_seed_{seed}.npy`

For example: `100_cells_3000_points_noise_0.1_seed_0.npy`

## Data Structure

Each `.npy` file should contain a dictionary with:
- `head_dir_rate`: Head direction cell firing rates (dependent on actual head direction)
- `head_dir_ind_rate`: Independent head direction cell rates (for H0)
- `grid_rate`: Grid cell firing rates
- `pos`: Position data (x, y coordinates)
- `head_direction`: Head direction in radians

## Generating Data

See the RatInABox library for generating simulation data:
https://github.com/RatInABox-Lab/RatInABox/

Also see the kernel-ci-testing repository for the original implementation:
https://github.com/romanpogodin/kernel-ci-testing
