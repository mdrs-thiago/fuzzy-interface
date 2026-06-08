import torch
import numpy as np
from fuzzy_torch import FuzzyModule, FuzzyVariable, FuzzyRule, TriangularMF
from fuzzy_torch.defuzz import Centroid

def setup_tank_fis():
    e = FuzzyVariable("Error", -10, 10)
    e.add_term("Negative", TriangularMF(-10, -10, 0))
    e.add_term("Zero", TriangularMF(-2, 0, 2))
    e.add_term("Positive", TriangularMF(0, 10, 10))

    v = FuzzyVariable("Valve", 0, 1)
    v.add_term("Closed", TriangularMF(0, 0, 0.4))
    v.add_term("Half", TriangularMF(0.2, 0.5, 0.8))
    v.add_term("Open", TriangularMF(0.6, 1, 1))

    rules = [
        FuzzyRule([(e, "Positive")], (v, "Open")),
        FuzzyRule([(e, "Zero")], (v, "Half")),
        FuzzyRule([(e, "Negative")], (v, "Closed"))
    ]
    return FuzzyModule(inputs=[e], output=v, rules=rules, defuzz_method=Centroid(0, 1))

def simulate_tank():
    print("Simulating Tank...")
    fis = setup_tank_fis()
    dt = 0.05
    A = 1.0
    a = 0.015  # Smaller leak for easier control
    g = 9.81
    K_in = 1.2 # Higher inflow for speed
    
    level = 2.0
    setpoint = 8.0
    history = []
    
    for _ in range(400):
        error = setpoint - level
        inp = {"Error": torch.tensor([error])}
        valve = fis(inp).item()
        q_out = a * np.sqrt(2 * g * max(0, level))
        q_in = valve * K_in
        level += (q_in - q_out) * dt / A
        level = max(0, min(10, level))
        history.append((level, valve))
    return history

def simulate_crane():
    print("Simulating Crane...")
    # Inputs: Error, Velocity, Angle, AngularVelocity
    # For a stable swing, we MUST have derivative terms (D)
    pos_err = FuzzyVariable("PosErr", -10, 10)
    pos_err.add_term("Neg", TriangularMF(-10, -10, 0))
    pos_err.add_term("Zero", TriangularMF(-1, 0, 1))
    pos_err.add_term("Pos", TriangularMF(0, 10, 10))
    
    vel = FuzzyVariable("Vel", -5, 5)
    vel.add_term("Neg", TriangularMF(-5, -5, 0))
    vel.add_term("Zero", TriangularMF(-0.5, 0, 0.5))
    vel.add_term("Pos", TriangularMF(0, 5, 5))
    
    angle = FuzzyVariable("Angle", -1, 1)
    angle.add_term("Neg", TriangularMF(-1, -1, 0))
    angle.add_term("Zero", TriangularMF(-0.1, 0, 0.1))
    angle.add_term("Pos", TriangularMF(0, 1, 1))

    w = FuzzyVariable("W", -2, 2)
    w.add_term("Neg", TriangularMF(-2, -2, 0))
    w.add_term("Zero", TriangularMF(-0.2, 0, 0.2))
    w.add_term("Pos", TriangularMF(0, 2, 2))
    
    force = FuzzyVariable("Force", -30, 30)
    force.add_term("Left", TriangularMF(-30, -30, 0))
    force.add_term("Zero", TriangularMF(-2, 0, 2))
    force.add_term("Right", TriangularMF(0, 30, 30))
    
    rules = [
        # Base Position Control
        FuzzyRule([(pos_err, "Pos")], (force, "Right")),
        FuzzyRule([(pos_err, "Neg")], (force, "Left")),
        # Damping (Anti-Overshoot)
        FuzzyRule([(vel, "Pos")], (force, "Left")),
        FuzzyRule([(vel, "Neg")], (force, "Right")),
        # Anti-Sway
        FuzzyRule([(angle, "Pos")], (force, "Left")),
        FuzzyRule([(angle, "Neg")], (force, "Right")),
        # Swing Damping
        FuzzyRule([(w, "Pos")], (force, "Left")),
        FuzzyRule([(w, "Neg")], (force, "Right"))
    ]
    
    fis = FuzzyModule(inputs=[pos_err, vel, angle, w], output=force, rules=rules, defuzz_method=Centroid(-30, 30))
    
    M, m, L, g = 2.0, 0.5, 2.0, 9.81
    x, v, theta, omega = 0.0, 0.0, 0.2, 0.0
    target = 5.0
    dt = 0.01

    history = []
    for _ in range(1000):
        # FIS inputs
        inp = {
            "PosErr": torch.tensor([target - x]),
            "Vel": torch.tensor([v]),
            "Angle": torch.tensor([theta]),
            "W": torch.tensor([omega])
        }
        u = fis(inp).item()
        
        # Physics
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        temp = (u + m*L*omega**2*sin_t) / (M + m)
        theta_acc = (g*sin_t - cos_t*temp) / (L*(4/3 - m*cos_t**2 / (M + m)))
        x_acc = temp - m*L*theta_acc*cos_t / (M + m)
        
        v += x_acc * dt
        x += v * dt
        omega += theta_acc * dt
        theta += omega * dt
        
        history.append((x, theta))
    return history

if __name__ == "__main__":
    tank_data = simulate_tank()
    print("\nTank Final State (Level/Setpoint):", tank_data[-1][0], "/ 8.0")
    
    crane_data = simulate_crane()
    print("Crane Final State (Pos/Target):", crane_data[-1][0], "/ 5.0")
    print("Crane Final Angle:", crane_data[-1][1])

    # Simple text plot for Tank
    print("\nTank Level Convergence:")
    for i in range(0, len(tank_data), 40):
        level = tank_data[i][0]
        bar = "#" * int(level * 4)
        print(f"T{i:3}: {level:5.2f} {bar}")

    # Simple text plot for Crane
    print("\nCrane Position Convergence:")
    for i in range(0, len(crane_data), 100):
        pos = crane_data[i][0]
        bar = " " * int(pos * 4) + "*"
        print(f"C{i:3}: {pos:5.2f} {bar}")
