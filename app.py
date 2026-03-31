import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🧿 PARAKLETOS — Physics Engine")

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
# POTENTIAL FIELD (GRAVITY-LIKE)
# -----------------------
def potential(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-5
    return -mass_strength / r

def grad_potential(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-5
    return (mass_strength * x / r**3,
            mass_strength * y / r**3,
            mass_strength * z / r**3)

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

    # ADD GRAVITY EFFECT
    gx,gy,gz = grad_potential(x,y,z)

    # PROJECT GRADIENT INTO PARAM SPACE (approx)
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

entropy_log = []

# -----------------------
# SIMULATION
# -----------------------
for _ in range(steps):
    u,v, fu, fv = step(u,v)

    # ENTROPY (gradient magnitude)
    entropy = np.sqrt(fu**2 + fv**2)
    entropy_log.append(np.mean(entropy))

# -----------------------
# FINAL POSITIONS
# -----------------------
x,y,z = xyz(u,v)

# -----------------------
# VISUAL
# -----------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

# surface
res=80
U = np.linspace(0,2*np.pi,res)
V = np.linspace(0,2*np.pi,res)
U,V = np.meshgrid(U,V)
X,Y,Z = xyz(U,V)

ax.plot_surface(X,Y,Z,alpha=0.1)

# COLOR BY ENTROPY
entropy = np.sqrt(fu**2 + fv**2) * entropy_scale
sc = ax.scatter(x,y,z,c=entropy,cmap='inferno',s=8)

ax.set_box_aspect([1,1,1])
fig.colorbar(sc, label="Entropy")

st.pyplot(fig)

# -----------------------
# METRICS
# -----------------------
st.subheader("Physics Metrics")

st.write("Mean Entropy:", float(np.mean(entropy)))
st.write("Entropy Stability:", float(np.std(entropy_log)))

if mass_strength > 0:
    st.write("Gravity Field: ACTIVE")
else:
    st.write("Gravity Field: OFF")
