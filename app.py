import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🧿 PARAKLETOS — Cosmic Engine")

# =========================
# CONTROLS
# =========================
col1, col2 = st.columns(2)

with col1:
    mass_strength = st.slider("Mass Strength", 0.1, 10.0, 2.5)
    dt = st.slider("Time Step", 0.001, 0.1, 0.02)

with col2:
    steps = st.slider("Steps", 200, 1200, 600)
    N = st.slider("Particles", 50, 300, 180)

orbit_mode = st.toggle("Orbit Mode")
black_hole_mode = st.toggle("Black Hole Core")
toroid_mode = st.toggle("Toroidal Manifold")
galaxy_mode = st.toggle("Galaxy Structure")
trail_mode = st.toggle("Trails")
ai_stabilizer = st.toggle("AI Stabilizer")

rs = st.slider("Event Horizon", 0.05, 1.0, 0.25)

# =========================
# ENGINE
# =========================
class Engine:
    def __init__(self, G=1.0):
        self.G = G

    def gravity(self, pos, masses):
        acc = np.zeros_like(pos)
        for i in range(len(pos)):
            diff = pos - pos[i]
            dist = np.linalg.norm(diff, axis=1) + 1e-5
            force = (self.G * masses[:,None] * diff) / (dist[:,None]**3)
            acc[i] = np.sum(force, axis=0)
        return acc

    def black_hole(self, pos, vel, rs):
        r = np.linalg.norm(pos, axis=1)
        mask = r < rs
        vel[mask] *= 0.1
        return vel

engine = Engine()

# =========================
# INITIAL CONDITIONS
# =========================
pos = np.random.randn(N,3)
vel = np.random.randn(N,3)*0.1
masses = np.ones(N)*mass_strength

# ORBIT MODE
if orbit_mode:
    pos[0] = np.array([0,0,0])
    masses[0] = 100

    for i in range(1,N):
        r = np.linalg.norm(pos[i])
        v_mag = np.sqrt(engine.G*masses[0]/(r+1e-5))
        direction = np.cross(pos[i],[0,0,1])
        direction /= np.linalg.norm(direction)+1e-5
        vel[i] = direction*v_mag

# GALAXY MODE
if galaxy_mode:
    for i in range(N):
        angle = np.random.uniform(0,2*np.pi)
        radius = np.random.uniform(0.5,5)

        pos[i] = np.array([
            radius*np.cos(angle),
            radius*np.sin(angle),
            np.random.normal(0,0.15)
        ])

        vel[i] = np.array([
            -np.sin(angle),
            np.cos(angle),
            0
        ])*0.8

# TORUS
def project_torus(pos,R=3.0,r=1.2):
    xy = np.sqrt(pos[:,0]**2+pos[:,1]**2)
    u = np.arctan2(pos[:,1],pos[:,0])
    v = np.arctan2(pos[:,2],xy-R)

    x = (R + r*np.cos(v))*np.cos(u)
    y = (R + r*np.cos(v))*np.sin(u)
    z = r*np.sin(v)

    return np.stack([x,y,z],axis=1)

# =========================
# SIMULATION
# =========================
plot = st.empty()

energy_log=[]
entropy_log=[]
history=[]

for step in range(steps):

    acc = engine.gravity(pos,masses)

    vel += acc*dt
    pos += vel*dt

    if black_hole_mode:
        vel = engine.black_hole(pos,vel,rs)

    if toroid_mode:
        pos = project_torus(pos)

    energy = 0.5*np.sum(vel**2,axis=1)
    entropy = energy

    energy_log.append(np.mean(energy))
    entropy_log.append(np.mean(entropy))

    # AI STABILIZER
    if ai_stabilizer and len(energy_log)>50:
        std = np.std(energy_log[-50:])
        if std>2:
            dt*=0.95
        elif std<0.2:
            dt*=1.02

    if trail_mode:
        history.append(pos.copy())
        if len(history)>8:
            history.pop(0)

    # =========================
    # VISUAL (INTENSE)
    # =========================
    if step%4==0:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')

        # glow layers
        for size,alpha in [(20,0.02),(10,0.05)]:
            ax.scatter(
                pos[:,0],pos[:,1],pos[:,2],
                c=energy,
                cmap='plasma',
                s=size,
                alpha=alpha
            )

        # main particles
        sc = ax.scatter(
            pos[:,0],pos[:,1],pos[:,2],
            c=energy,
            cmap='inferno',
            s=6
        )

        # trails
        if trail_mode:
            for past in history:
                ax.plot(past[:,0],past[:,1],past[:,2],alpha=0.05,color='cyan')

        # black hole visual
        if black_hole_mode:
            ax.scatter(0,0,0,s=300,color='black')

        ax.set_xlim(-7,7)
        ax.set_ylim(-7,7)
        ax.set_zlim(-7,7)

        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        ax.set_title(f"Step {step}",color="white")

        plot.pyplot(fig)
        plt.close(fig)

# =========================
# METRICS
# =========================
st.subheader("📊 System Metrics")

col1,col2 = st.columns(2)

with col1:
    st.metric("Energy", f"{np.mean(energy_log):.4f}")
    st.metric("Energy Stability", f"{np.std(energy_log):.4f}")

with col2:
    st.metric("Entropy", f"{np.mean(entropy_log):.4f}")
    st.metric("Entropy Stability", f"{np.std(entropy_log):.4f}")

st.success("System Running — Adaptive Physics Active")
