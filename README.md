# ABS-Simulator
This project simulates the effect of an Anti-lock Braking System (ABS) on a vehicle using a physics-based model with longitudinal dynamics, tire slip modeling, and traction force prediction using a Pacejka tire model.

- Models car and wheel dynamics with and without ABS
- Implements Pacejka tire model to simulate traction-slip curve
- Real-time feedback controller (PID + feedforward) for ABS torque modulation
- Visual comparison of:
- Car vs wheel velocity with and without ABS
- Stopping distance
- Slip ratio and traction force animation
- Event-triggered simulation stop when the car reaches zero velocity

Braking Torques:
- Applies constant braking torque when ABS is OFF
- Applies optimal braking torque using PID and Feedforward when ABS is ON
