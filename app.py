import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🧿 PARAKLETOS — Integrated Physics System")

# -----------------------
# CONTROLS
# -----------------------
R = st.slider("R", 0.5, 2.0, 1.0)
r = st.slider("r", 1.0, 3.0, 1.5)
dt = st.slider("dt", 0.01, 0.1, 0.03)
steps = st.slider("Steps", 100, 2000, 500)

mass_strength = st.slider("Mass Strength", 0.0, 5.0, 1.5)
entropy_scale = st.slider("Entropy Sensitivity", 0.1, 3.0, 1.0)

invert = st.toggle("Invert Field")
auto_stabilize = st.toggle("AI Stabilizer")

# -----------------------
# GEOMETRY
# -----------------------
def xyz(u,v):
    x = (R + r*np.cos(v))*np.cos(u)
    y = (R + r*np.cos(v))*np.sin(u)
    z = r*np.sin(v)
    return x,y,z

def metric(u,v):
    return (R + r*np.cos(v)), r

# -----------------------
# POTENTIAL (GRAVITY-LIKE)
# -----------------------
def grad_potential(x,y,z):
    r_mag = np.sqrt(x**2 + y**2 + z**2) + 1e-5
    return (mass_strength * x / r_mag**3,
            mass_strength * y / r_mag**3,
            mass_strength * z / r_mag**3)

# -----------------------
# BASE FLOW
# -----------------------
def flow(u,v):
    hu,hv = metric(u,v)

    dpsi_du = np.cos(u)*np.cos(v)
    dpsi_dv = -np.sin(u)*np.sin(v)

    Fu = (1/hu)*dpsi_dv
    Fv = -(1/hv)*dpsi_du

    return Fu,Fv

# -----------------------
# STEP
# -----------------------
def step(u,v):
    x,y,z = xyz(u,v)

    Fu,Fv = flow(u,v)

    gx,gy,gz = grad_potential(x,y,z)

    Fu += gx * 0.1
    Fv += gy * 0.1

    if invert:
        Fu, Fv = -Fu, -Fv

    u = (u + dt*Fu) % (2*np.pi)
    v = (v + dt*Fv) % (2*np.pi)

    return u,v, Fu, Fv

# -----------------------
# INIT
# -----------------------
N = 400
u = np.random.rand(N)*2*np.pi
v = np.random.rand(N)*2*np.pi

energy_log = []
entropy_log = []

# -----------------------
# SIMULATION
# -----------------------
for _ in range(steps):
    u,v, fu, fv = step(u,v)

    energy = np.sqrt(fu**2 + fv**2)
    entropy = energy  # proxy

    energy_log.append(np.mean(energy))
    entropy_log.append(np.mean(entropy))

# -----------------------
# AI STABILIZER
# -----------------------
energy_std = np.std(energy_log)
entropy_std = np.std(entropy_log)

if auto_stabilize:
    if energy_std > 0.5:
        mass_strength *= 0.9
    if entropy_std > 0.5:
        entropy_scale *= 0.9

# -----------------------
# FINAL POSITIONS
# -----------------------
x,y,z = xyz(u,v)

# -----------------------
# VISUAL
# -----------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

res=80
U = np.linspace(0,2*np.pi,res)
V = np.linspace(0,2*np.pi,res)
U,V = np.meshgrid(U,V)
X,Y,Z = xyz(U,V)

ax.plot_surface(X,Y,Z,alpha=0.1)

color_field = np.sqrt(fu**2 + fv**2) * entropy_scale

sc = ax.scatter(x,y,z,c=color_field,cmap='inferno',s=8)

ax.set_box_aspect([1,1,1])
fig.colorbar(sc, label="Energy / Entropy")

st.pyplot(fig)

# -----------------------
# METRICS
# -----------------------
st.subheader("System Metrics")

st.write("Mean Energy:", float(np.mean(energy_log)))
st.write("Energy Stability:", float(energy_std))

st.write("Mean Entropy:", float(np.mean(entropy_log)))
st.write("Entropy Stability:", float(entropy_std))

if auto_stabilize:
    st.success("AI Stabilizer: ACTIVE")
else:
    st.info("AI Stabilizer: OFF")
