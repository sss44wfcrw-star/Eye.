import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🧿 PARAKLETOS — Live System")

R = st.slider("R", 0.5, 2.0, 1.0)
r = st.slider("r", 1.0, 3.0, 1.5)
dt = st.slider("dt", 0.01, 0.1, 0.03)
steps = st.slider("Steps", 100, 2000, 500)

def xyz(u,v):
    x = (R + r*np.cos(v))*np.cos(u)
    y = (R + r*np.cos(v))*np.sin(u)
    z = r*np.sin(v)
    return x,y,z

def metric(u,v):
    return (R + r*np.cos(v)), r

def flow(u,v):
    hu,hv = metric(u,v)
    dpsi_du = np.cos(u)*np.cos(v)
    dpsi_dv = -np.sin(u)*np.sin(v)
    Fu = (1/hu)*dpsi_dv
    Fv = -(1/hv)*dpsi_du
    return Fu,Fv

def step(u,v):
    fu1,fv1 = flow(u,v)
    u_mid = (u + 0.5*dt*fu1) % (2*np.pi)
    v_mid = (v + 0.5*dt*fv1) % (2*np.pi)
    fu2,fv2 = flow(u_mid,v_mid)
    u = (u + dt*fu2) % (2*np.pi)
    v = (v + dt*fv2) % (2*np.pi)
    return u,v

N = 400
u = np.random.rand(N)*2*np.pi
v = np.random.rand(N)*2*np.pi

for _ in range(steps):
    u,v = step(u,v)

x,y,z = xyz(u,v)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

res=80
U = np.linspace(0,2*np.pi,res)
V = np.linspace(0,2*np.pi,res)
U,V = np.meshgrid(U,V)

X,Y,Z = xyz(U,V)
ax.plot_surface(X,Y,Z,alpha=0.15)
ax.scatter(x,y,z,s=5)

ax.set_box_aspect([1,1,1])

st.pyplot(fig)
