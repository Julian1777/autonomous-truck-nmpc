# Autonomous Truck NMPC

An early-stage project exploring autonomous truck control on a multi-lane highway.

The goal is to build a 2D simulation in which a truck and trailer can plan and
follow safe, efficient trajectories using nonlinear model predictive control
(NMPC). The controller is intended to evaluate multiple possible trajectories
and choose the one with the lowest cost while considering the truck's dynamics,
lane position, speed, and surrounding traffic.

## Planned Features

- 2D multi-lane highway simulation
- Nonlinear truck and trailer dynamics
- NMPC-based steering and acceleration control
- Cost-efficient lane selection
- Intelligent traffic with other vehicles
- Trajectory and controller visualisation

## Current Status

This project is still under development. The current code contains an initial
truck/trailer dynamics model and the beginning of an NMPC controller. The
simulation, traffic behaviour, lane selection, and visualisation are not
complete yet.

## Project Files

- `dynamics.py` - Truck/trailer dynamics and Runge-Kutta integration
- `mpc_controller.py` - Initial CasADi-based NMPC controller

## Long-Term Goal

The long-term goal is a realistic autonomous highway scenario where the truck
can select an efficient lane, respond to other vehicles, and follow smooth,
safe trajectories while respecting its physical constraints.

The project is inspired by [Autonomous-Truck-Sim](https://github.com/BorveErik/Autonomous-Truck-Sim/).
