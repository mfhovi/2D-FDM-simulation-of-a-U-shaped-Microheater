import numpy as np
import matplotlib.pyplot as plt

# ----------------- PARAMETERS -----------------
Lx = 10.0  # mm
Ly = 10.0  # mm
dx = dy = 0.1  # grid spacing in mm, uniform grid for simplicity
nx = int(Lx/dx) + 1
ny = int(Ly/dy) + 1

T_boundary = 25.0 # boundary temperature in °C
T_heater = 185.0 # heater temperature in °C {variable Input}
tol = 1e-6 # convergence tolerance
max_iter = 100000  # maximum iterations

# ------------ GRID INITIALIZATION ----
T = np.full((ny, nx), T_boundary)

a = 1.0 / dx**2 # coefficients for FDM
b = 1.0 / dy**2 # coefficients for FDM
denom = 2 * (a + b) # denominator in update formula

# ---------- MICROHEATER DEFINITION --
heater_mask = np.zeros((ny, nx), dtype=bool)

def coord_to_index(x, y):
    return int(round(y/dy)), int(round(x/dx))

x_left, x_right = 4.5, 5.5
y_bottom, y_top = 4.5, 5.5
# mask is true where heater is located
# Horizontal bottom line
for x in np.arange(x_left, x_right+dx, dx):
    j, i = coord_to_index(x, y_bottom)
    heater_mask[j, i] = True

# Vertical left leg
for y in np.arange(y_bottom, y_top+dy, dy):
    j, i = coord_to_index(x_left, y)
    heater_mask[j, i] = True

# Vertical right leg
for y in np.arange(y_bottom, y_top+dy, dy):
    j, i = coord_to_index(x_right, y)
    heater_mask[j, i] = True

# Setting  heater temperature
T[heater_mask] = T_heater

# -------- ITERATIVE SOLVER ----
for it in range(max_iter):
    max_diff = 0.0
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if heater_mask[j, i]: # Skip heater cells
                continue
            old_T = T[j, i] # Store old value for convergence check
            T[j, i] = (a*(T[j, i+1] + T[j, i-1]) + b*(T[j+1, i] + T[j-1, i])) / denom # Update formula
            max_diff = max(max_diff, abs(T[j, i] - old_T)) # Update max difference
    if max_diff < tol: # Check convergence
        print(f'Converged in {it} iterations')
        break
else:
    print("Did not converge!")

# ------- Getting simulation Data -----
inside_mask = np.zeros((ny, nx), dtype=bool) # Mask for the rectanglular region inside U excluding heater
x_inside = (np.arange(nx)*dx >= x_left+dx) & (np.arange(nx)*dx <= x_right-dx)
y_inside = (np.arange(ny)*dy >= y_bottom+dy) & (np.arange(ny)*dy <= y_top-dy)
for j in range(ny): 
    for i in range(nx):
        if x_inside[i] and y_inside[j] and not heater_mask[j, i]:
            inside_mask[j, i] = True
# Average temperature inside the rectangular region
avg_temp_inside = np.mean(T[inside_mask])

# Temperature 2 mm below the bottom of U
y_query = y_bottom - 2.0
x_query = (x_left + x_right) / 2.0
jq, iq = coord_to_index(x_query, y_query)
temp_below = T[jq, iq]

print(f"Average temperature inside U (excluding heater): {avg_temp_inside:.2f} °C")
print(f"Temperature at 2 mm below bottom of U (measurable temperature): {temp_below:.2f} °C")

# ----------------- VISUALIZATION -----------------
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, T, levels=50, cmap='jet')
plt.colorbar(cp, label='Temperature (°C)')
plt.contour(X, Y, T, levels=20, colors='black', linewidths=0.5)
plt.contour(X, Y, heater_mask, levels=[0.5], colors='white', linewidths=2)

# Annotate average region
inside_y, inside_x = np.where(inside_mask)
plt.scatter(inside_x*dx, inside_y*dy, s=1, color='cyan', alpha=0.5, label='Inside region')

# Annotate point 2mm below heater
plt.plot(x_query, y_query, 'ro', label='2 mm below heater')

plt.legend(loc='upper right')
plt.title('2D Steady-State Temperature Field with U-shaped Microheater')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.show()
