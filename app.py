import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="PARAKLETOS", layout="wide")

st.title("🧿 PARAKLETOS — Integrated Physics System")

# =========================
# CONTROLS
# =========================
col1, col2 = st.columns(2)

with col1:
    mass_strength = st.slider("Mass Strength", 0.1, 5.0, 1.5)
    dt = st.slider("Time Step (dt)", 0.001, 0.1, 0.03)

with col2:
    steps = st.slider("Steps", 100, 2000, 500)
    num_particles = st.slider("Particles", 50, 300, 150)

orbit_mode = st.toggle("Orbit Mode")
black_hole_mode = st.toggle("Black Hole Mode")
ai_stabilizer = st.toggle("AI Stabilizer")

rs = st.slider("Event Horizon (Black Hole Radius)", 0.05, 1.0, 0.2)

# =========================
# PHYSICS ENGINE
# =========================
class PhysicsEngine:
    def __init__(self, G=1.0):
        self.G = G

    def gravity_multi(self, pos, masses):
        acc = np.zeros_like(pos)

        for i in range(len(pos)):
            for j in range(len(pos)):
                if i == j:
                    continue
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec) + 1e-5
                acc[i] += self.G * masses[j] * r_vec / (r**3)

        return acc

    def black_hole_effect(self, pos, velocities, rs):
        for i in range(len(pos)):
            r = np.linalg.norm(pos[i])
            if r < rs:
                velocities[i] *= 0.2
        return velocities


# =========================
# INITIALIZE SYSTEM
# =========================
engine = PhysicsEngine()

pos = np.random.randn(num_particles, 3)
vel = np.random.randn(num_particles, 3) * 0.1
masses = np.ones(num_particles) * mass_strength

# ORBIT MODE
if orbit_mode:
    pos[0] = np.array([0, 0, 0])
    masses[0] = 50.0

    for i in range(1, num_particles):
        r = np.linalg.norm(pos[i])
        speed = np.sqrt(engine.G * masses[0] / (r + 1e-5))

        direction = np.cross(pos[i], [0, 0, 1])
        direction = direction / (np.linalg.norm(direction) + 1e-5)

        vel[i] = direction * speed


# =========================
# SIMULATION
# =========================
energy_log = []
entropy_log = []

placeholder = st.empty()

for step in range(steps):
    acc = engine.gravity_multi(pos, masses)

    vel += acc * dt
    pos += vel * dt

    if black_hole_mode:
        vel = engine.black_hole_effect(pos, vel, rs)

    energy = 0.5 * np.sum(vel**2, axis=1)
    entropy = energy

    energy_log.append(np.mean(energy))
    entropy_log.append(np.mean(entropy))

    # =========================
    # AI STABILIZER
    # =========================
    if ai_stabilizer and len(energy_log) > 50:
        energy_std = np.std(energy_log[-50:])

        if energy_std > 1.5:
            mass_strength *= 0.97
            dt *= 0.95
        elif energy_std < 0.2:
            mass_strength *= 1.02

    # =========================
    # LIVE VISUAL
    # =========================
    if step % 5 == 0:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=5)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)

        ax.set_title(f"Step {step}")

        placeholder.pyplot(fig)
        plt.close(fig)


# =========================
# METRICS
# =========================
st.subheader("System Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Mean Energy", f"{np.mean(energy_log):.4f}")
    st.metric("Energy Stability", f"{np.std(energy_log):.4f}")

with col2:
    st.metric("Mean Entropy", f"{np.mean(entropy_log):.4f}")
    st.metric("Entropy Stability", f"{np.std(entropy_log):.4f}")

st.write(f"AI Stabilizer: {'ON' if ai_stabilizer else 'OFF'}")
