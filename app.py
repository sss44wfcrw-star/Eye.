import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🧿 PARAKLETOS — Physics System")

# -----------------------
# CONTROLS
# -----------------------
R = st.slider("R", 0.5, 2.0, 1.0)
r = st.slider("r", 1.0, 3.0, 1.5)
dt = st.slider("dt", 0.01, 0.1, 0.03)
steps = st.slider("Steps", 100, 2000, 500)
invert = st.toggle("Invert Field (F → -F)")

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
# FIELD
# -----------------------
def flow(u,v):
    hu,hv = metric(u,v)
    dpsi_du = np.cos(u)*np.cos(v)
    dpsi_dv = -np.sin(u)*np.sin(v)
    Fu = (1/hu)*dpsi_dv
    Fv = -(1/hv)*dpsi_du

    if invert:
        Fu, Fv = -Fu, -Fv

    return Fu,Fv

# -----------------------
# STEP (MIDPOINT)
# -----------------------
def step(u,v):
    fu1,fv1 = flow(u,v)
    u_mid = (u + 0.5*dt*fu1) % (2*np.pi)
    v_mid = (v + 0.5*dt*fv1) % (2*np.pi)

    fu2,fv2 = flow(u_mid,v_mid)

    u = (u + dt*fu2) % (2*np.pi)
    v = (v + dt*fv2) % (2*np.pi)

    return u,v, fu2, fv2

# -----------------------
# INIT
# -----------------------
N = 400
u = np.random.rand(N)*2*np.pi
v = np.random.rand(N)*2*np.pi

energy_log = []

# -----------------------
# SIMULATION
# -----------------------
for _ in range(steps):
    u,v, fu, fv = step(u,v)

    # ENERGY (velocity magnitude)
    energy = np.sqrt(fu**2 + fv**2)
    energy_log.append(np.mean(energy))

# -----------------------
# FINAL POSITIONS
# -----------------------
x,y,z = xyz(u,v)

# -----------------------
# PLOT
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

# ENERGY COLOR
energy = np.sqrt(fu**2 + fv**2)
sc = ax.scatter(x,y,z,c=energy,cmap='plasma',s=8)

ax.set_box_aspect([1,1,1])
fig.colorbar(sc, label="Energy")

st.pyplot(fig)

# -----------------------
# METRICS
# -----------------------
st.subheader("System Metrics")

st.write("Average Energy:", float(np.mean(energy)))
st.write("Energy Stability (std):", float(np.std(energy_log)))

if invert:
    st.write("Field Mode: Inverted")
else:
    st.write("Field Mode: Normal")
