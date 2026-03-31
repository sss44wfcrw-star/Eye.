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
    mass_strength = st.slider("Mass Strength", 0.1, 10.0, 2.0)
    dt = st.slider("Time Step", 0.001, 0.1, 0.02)

with col2:
    steps = st.slider("Steps", 100, 2000, 500)
    N = st.slider("Particles", 50, 400, 200)

orbit_mode = st.toggle("Orbit Mode")
black_hole_mode = st.toggle("Black Hole Mode")
ai_stabilizer = st.toggle("AI Stabilizer")
toroid_mode = st.toggle("Toroidal Field Mode")
galaxy_mode = st.toggle("Galaxy Mode")

rs = st.slider("Event Horizon Radius", 0.05, 1.0, 0.2)

# =========================
# ENGINE
# =========================
class Engine:
    def __init__(self, G=1.0):
        self.G = G

    def gravity(self, pos, masses):
        acc = np.zeros_like(pos)
        for i in range(len(pos)):
            for j in range(len(pos)):
                if i == j:
                    continue
                r_vec = pos[j] - pos[i]
                r = np.linalg.norm(r_vec) + 1e-5
                acc[i] += self.G * masses[j] * r_vec / (r**3)
        return acc

    def black_hole(self, pos, vel, rs):
        for i in range(len(pos)):
            if np.linalg.norm(pos[i]) < rs:
                vel[i] *= 0.1
        return vel

engine = Engine()

# =========================
# INITIAL CONDITIONS
# =========================
pos = np.random.randn(N, 3)
vel = np.random.randn(N, 3) * 0.1
masses = np.ones(N) * mass_strength

# ORBIT MODE
if orbit_mode:
    pos[0] = np.array([0,0,0])
    masses[0] = 80.0

    for i in range(1, N):
        r = np.linalg.norm(pos[i])
        speed = np.sqrt(engine.G * masses[0] / (r+1e-5))
        direction = np.cross(pos[i], [0,0,1])
        direction /= (np.linalg.norm(direction)+1e-5)
        vel[i] = direction * speed

# GALAXY MODE
if galaxy_mode:
    for i in range(N):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0.5, 3.0)

        pos[i] = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.random.normal(0, 0.1)
        ])

        vel[i] = np.array([
            -np.sin(angle),
            np.cos(angle),
            0
        ]) * 0.5

# =========================
# TOROID PROJECTION
# =========================
def project_torus(pos, R=2.0, r=1.0):
    new_pos = []
    for p in pos:
        u = np.arctan2(p[1], p[0])
        v = np.arctan2(p[2], np.sqrt(p[0]**2 + p[1]**2) - R)

        x = (R + r*np.cos(v))*np.cos(u)
        y = (R + r*np.cos(v))*np.sin(u)
        z = r*np.sin(v)

        new_pos.append([x,y,z])

    return np.array(new_pos)

# =========================
# SIMULATION
# =========================
energy_log = []
entropy_log = []

plot = st.empty()

for step in range(steps):

    acc = engine.gravity(pos, masses)

    vel += acc * dt
    pos += vel * dt

    if black_hole_mode:
        vel = engine.black_hole(pos, vel, rs)

    if toroid_mode:
        pos = project_torus(pos)

    # ENERGY / ENTROPY
    energy = 0.5 * np.sum(vel**2, axis=1)
    entropy = energy

    energy_log.append(np.mean(energy))
    entropy_log.append(np.mean(entropy))

    # =========================
    # AI STABILIZER
    # =========================
    if ai_stabilizer and len(energy_log) > 50:
        std = np.std(energy_log[-50:])
        if std > 2.0:
            mass_strength *= 0.95
            dt *= 0.95
        elif std < 0.2:
            mass_strength *= 1.05

    # =========================
    # LIVE RENDER
    # =========================
    if step % 5 == 0:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')

        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=6)

        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-5,5)

        ax.set_title(f"Step {step}")

        plot.pyplot(fig)
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
