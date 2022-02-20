from distutils.log import debug
import numpy as np
import sys

import taichi as ti

ti.init(arch=ti.cpu)  # Try to run on GPU
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 100000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 0.1e-4 / quality   #1e-4
elapsed_time =ti.field(dtype=float, shape=())
n_substeps =ti.field(dtype=int, shape=())
indice_temoin =ti.field(dtype=int, shape=())
#p_vol, p_rho = (dx * 0.5)**2, 1
p_vol, p_rho = (dx)**2, 1
p_mass = p_vol * p_rho
#E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio  (original!)
E, nu = 500, 0.4  # Young's modulus and Poisson's ratio  0.1e5, 0.2
g = 30

# if len(sys.argv) > 1:
#     n_particles = int(sys.argv[1])
# print("n_particles = ", n_particles)
if len(sys.argv) > 1:
    E = float(sys.argv[1])
print("E = ", E)
if len(sys.argv) > 2:
    nu = float(sys.argv[2])
print("nu = ", nu) 
if len(sys.argv) > 3:
    g = float(sys.argv[3])
print("g = ", g) 
if len(sys.argv) == 1:   
    print("pas de parametres")


fichier_res = "MPM-"  + "E" + str(int(E)) + "-nu" + "{:.2f}".format(nu) + "-g" + str(int(g))+ ".res"
fich = open(fichier_res, "w" )
print("writing to ", fichier_res)


mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float,
                    shape=n_particles)  # deformation gradient                 
material = ti.field(dtype=int, shape=n_particles)  # material id
core_cortex = ti.field(dtype=int, shape=n_particles)  # 0 for core, 1 for cortex
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float,
                         shape=(n_grid, n_grid))  # grid node momentum/velocity

grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass

# geometrical dimensions (L=1 by default)
length_beam = 0.6; 
width_beam = 0.10
point_bas = 0.8  
indice_bas = int(point_bas * n_grid)
indice_haut = int(indice_bas + width_beam * n_grid)
gamma = 0.9999


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) +
                dt * C[p]) @ F[p]  # deformation gradient update            #...EQ. 181 PAGE 42
        h = ti.exp(
            10 *                                                            #...HARDENING=10
            (1.0 -
             Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        h = 1.0 # no change
        mu, la = mu_0 * h, lambda_0 * h       
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                              1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[
                p] == 0:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose(
            )  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose(
        ) + ti.Matrix.identity(float, 2) * la * J * (J - 1)                     #...C'EST J*SIGMA..PAGE 20 et 18 et 19
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]                                         #...OUI EQ. 29
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)   #...OUI on rajoute dvi à vi calculé avec xp et Cp
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:        
        if grid_m[i, j] > 0:  # No need for epsilon here
            #if n_substeps[None] < 2:   print("masse du noeud = ",grid_m[i, j] )
            grid_v[i,
                   j] = (1 / grid_m[i, j]) * grid_v[i,
                                                    j]  # Momentum to velocity
            grid_v[i, j][1] -= dt * g  # gravity 50
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0
            
            # definition of beam
            if i < 3 and (j >= indice_bas and j <= indice_haut):
                grid_v[i, j][0] = 0
                grid_v[i, j][1] = 0
            #if n_substeps[None] < 2 and (grid_v[i, j][0] != 0.0 or grid_v[i, j][1] != 0.0):   
            #print("grid_v = ", grid_v[i, j][0], " ",grid_v[i, j][1] )
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(
                3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v * gamma
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
    old_elapsed_time = elapsed_time[None]
    elapsed_time[None] += dt
    n_substeps[None] += 1
    if int(elapsed_time[None]/0.1) > int(old_elapsed_time/0.1) :
        print('elapsed time = ',elapsed_time[None])
    

# beam
@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i]=[ti.random()*length_beam, ti.random()*width_beam + point_bas]
        if x[i][0] > 0.98 * length_beam and indice_temoin[None] == 0: 
            indice_temoin[None] = i
        material[i] = 1  # 0: fluid 1: jelly 2: snow        
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
    elapsed_time[None] = 0.0
    n_substeps[None] = 0



initialize()
#gui = ti.GUI("Taichi MLS-MPM-99", res=1024, background_color=0x112F41)
limit_range = 30 #int(max(30,2e-3 // dt))
print("limit_range = ",limit_range)
print("densité = ", n_particles/(width_beam*length_beam))
#while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT): 
while (elapsed_time[None] < 0.8):   
    for s in range(limit_range):   #range(int(2e-3 // dt))
        substep()
        if n_substeps[None]%400 == 0 :
            print("nb_steps = ", n_substeps[None], " indice_temoin =", indice_temoin[None], "particule_temoin =", x[indice_temoin[None]][0], " ",x[indice_temoin[None]][1] )
    # gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    # gui.show(
    # )  # Change to gui.show(f'{frame:06d}.png') to write images to disk

print("mu = ", mu_0 , "lambda = ", lambda_0)
print("nb appels à substeps = ", n_substeps[None])

#ecriture fichier
#fich = open("fichier_plot_beam_MPM.txt", "w" )
scatter_plot_MPM = []
liste_p = []
#x_numpy = x.to_numpy()

#print(x_numpy)
n_pas = 1024
pas = 1/n_pas
for p in range(n_particles) :
    indice_i = int(x[p][0]/pas)
    liste_p.append([indice_i, x[p][1]])

print("long de liste_p= ", len(liste_p))

for i in range(n_pas):
    ligne_v = [el[1] for el in liste_p if el[0]==i ]
    if len(ligne_v) > 0 :
        min_y = min(ligne_v )
        max_y = max(ligne_v )
        scatter_plot_MPM.extend([[i*pas, min_y], [i*pas, max_y]]) 

print("long de scatter_plot= ", len(scatter_plot_MPM))
for p in scatter_plot_MPM:
    fich.write(str(p[0]) + " " + str(p[1]) + "\n")
  
fich.close()

