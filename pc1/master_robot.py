"""
╔══════════════════════════════════════════════════════════════╗
║  ROBOT MAESTRO — TE3001B  Peg-in-Hole Teleoperado            ║
║  Simulación 3-DOF + Computed Torque + Graficación            ║
║  Prof. Alberto Muñoz — Computational Robotics Lab            ║
║  Tec de Monterrey, 2026                                      ║
╚══════════════════════════════════════════════════════════════╝

Ejecutar en la PC MAESTRO:
    python3 master_robot.py --slave-ip <IP_DEL_ESCLAVO>

Controles:
    W/S  → mover efector final en +y/-y
    A/D  → mover efector final en -x/+x
    Q/E  → abrir/cerrar pinza
    ESC  → salir y guardar datos
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # cambiar a 'Qt5Agg' si TkAgg falla
import matplotlib as mpl
mpl.rcParams['keymap.save'] = []
mpl.rcParams['keymap.quit'] = []
mpl.rcParams['keymap.fullscreen'] = []
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
import threading
import time
import json
import argparse

# ──────────────────────────────────────────────
# PARÁMETROS DEL ROBOT 3R
# ──────────────────────────────────────────────
L1, L2, L3 = 0.35, 0.30, 0.20  # longitudes eslabones [m]
M1, M2, M3 = 1.5, 1.0, 0.5     # masas de eslabones  [kg]
G_GRAV      = 9.81              # aceleración gravitacional [m/s^2]

# Ganancias del Computed Torque
KP = np.diag([120.0, 100.0, 80.0])   # rigidez articular
KV = np.diag([25.0,  20.0,  15.0])   # amortiguamiento

# Paso de integración
DT = 0.01    # 100 Hz

# ──────────────────────────────────────────────
# CINEMÁTICA DIRECTA 3R PLANAR
# ──────────────────────────────────────────────
def fk_3r(q):
    """
    Cinemática directa para robot 3R planar.

    Args:
        q (np.ndarray): Vector de ángulos articulares [q1, q2, q3] en radianes.

    Returns:
        tuple: (x, y) posición del efector final en metros.
    """
    q1, q2, q3 = q
    # Posiciones de cada articulación (útil para visualización)
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    x3 = x2 + L3 * np.cos(q1 + q2 + q3)
    y3 = y2 + L3 * np.sin(q1 + q2 + q3)
    return np.array([x3, y3])


def fk_3r_full(q):
    """
    Retorna las posiciones de todas las articulaciones y el EF.

    Returns:
        np.ndarray: Matriz (4x2) con posiciones [base, j1, j2, EF].
    """
    q1, q2, q3 = q
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
    p2 = p1 + np.array([L2*np.cos(q1+q2), L2*np.sin(q1+q2)])
    p3 = p2 + np.array([L3*np.cos(q1+q2+q3), L3*np.sin(q1+q2+q3)])
    return np.array([p0, p1, p2, p3])


# ──────────────────────────────────────────────
# JACOBIANO ANALÍTICO 3R PLANAR  (J ∈ R^{2×3})
# ──────────────────────────────────────────────
def jacobian_3r(q):
    """
    Jacobiano analítico del robot 3R planar.
    Relaciona velocidades articulares con velocidad cartesiana del EF:
        ẋ = J(q) q̇

    Args:
        q (np.ndarray): Vector de ángulos articulares [q1, q2, q3].

    Returns:
        np.ndarray: Matriz Jacobiana 2×3.
    """
    q1, q2, q3 = q
    s1   = np.sin(q1)
    s12  = np.sin(q1 + q2)
    s123 = np.sin(q1 + q2 + q3)
    c1   = np.cos(q1)
    c12  = np.cos(q1 + q2)
    c123 = np.cos(q1 + q2 + q3)

    J = np.array([
        [-L1*s1 - L2*s12 - L3*s123,
         -L2*s12 - L3*s123,
         -L3*s123],
        [ L1*c1 + L2*c12 + L3*c123,
          L2*c12 + L3*c123,
          L3*c123]
    ])
    return J


# ──────────────────────────────────────────────
# MODELO DINÁMICO SIMPLIFICADO (punto de masa)
# ──────────────────────────────────────────────
def inertia_matrix(q):
    """
    Matriz de inercia M(q) para robot 3R (modelo punto de masa al extremo).
    Propiedad: simétrica y positiva definida en todo el espacio de trabajo.

    Args:
        q (np.ndarray): Ángulos articulares.

    Returns:
        np.ndarray: Matriz de inercia 3×3.
    """
    q1, q2, q3 = q
    c2  = np.cos(q2)
    c3  = np.cos(q3)
    c23 = np.cos(q2 + q3)

    # Términos dominantes (modelo simplificado punto de masa)
    m11 = (M1*L1**2 + M2*(L1**2 + L2**2 + 2*L1*L2*c2) +
           M3*(L1**2 + L2**2 + L3**2 + 2*L1*L2*c2 +
               2*L1*L3*c23 + 2*L2*L3*c3))
    m12 = (M2*(L2**2 + L1*L2*c2) +
           M3*(L2**2 + L3**2 + L1*L2*c2 + L1*L3*c23 + 2*L2*L3*c3))
    m13 = M3*(L3**2 + L1*L3*c23 + L2*L3*c3)
    m22 = M2*L2**2 + M3*(L2**2 + L3**2 + 2*L2*L3*c3)
    m23 = M3*(L3**2 + L2*L3*c3)
    m33 = M3 * L3**2

    M = np.array([[m11, m12, m13],
                  [m12, m22, m23],
                  [m13, m23, m33]])
    return M


def coriolis_matrix(q, dq):
    """
    Matriz de Coriolis y fuerzas centrífugas C(q, q̇).
    Calculada via diferenciación numérica de M(q) (método Christoffel).

    Args:
        q  (np.ndarray): Ángulos articulares.
        dq (np.ndarray): Velocidades articulares.

    Returns:
        np.ndarray: Matriz C 3×3 tal que C*q̇ son las fuerzas de Coriolis.
    """
    eps = 1e-5
    n = len(q)
    M0 = inertia_matrix(q)
    C = np.zeros((n, n))

    # Símbolos de Christoffel (simplificados, solo términos dominantes)
    for k in range(n):
        qp = q.copy(); qp[k] += eps
        qm = q.copy(); qm[k] -= eps
        dM_dk = (inertia_matrix(qp) - inertia_matrix(qm)) / (2*eps)
        C += 0.5 * dM_dk * dq[k]

    return C


def gravity_vector(q):
    """
    Vector de par gravitacional g(q) para robot 3R planar.
    Asume que la gravedad actúa en dirección -y del plano.

    Args:
        q (np.ndarray): Ángulos articulares.

    Returns:
        np.ndarray: Vector de pares gravitacionales 3×1 [Nm].
    """
    q1, q2, q3 = q
    c1   = np.cos(q1)
    c12  = np.cos(q1 + q2)
    c123 = np.cos(q1 + q2 + q3)

    g1 = G_GRAV * ((M1+M2+M3)*L1*c1 + (M2+M3)*L2*c12 + M3*L3*c123)
    g2 = G_GRAV * ((M2+M3)*L2*c12 + M3*L3*c123)
    g3 = G_GRAV * M3 * L3 * c123
    return np.array([g1, g2, g3])


# ──────────────────────────────────────────────
# COMPUTED TORQUE CONTROLLER
# ──────────────────────────────────────────────
def computed_torque(q, dq, q_des, dq_des, ddq_des,
                    kp=KP, kv=KV, F_ext=None):
    """
    Ley de control de Computed Torque (cancelación exacta de no-linealidades).

    Control:  τ = M(q)·aₐ + C(q,q̇)·q̇ + g(q)
    donde:    aₐ = q̈ₐ + Kv(q̇ₐ - q̇) + Kp(qₐ - q)

    El sistema en lazo cerrado se reduce a: ë + Kv·ė + Kp·e = 0

    Args:
        q       (np.ndarray): Posición articular actual [rad].
        dq      (np.ndarray): Velocidad articular actual [rad/s].
        q_des   (np.ndarray): Posición articular deseada [rad].
        dq_des  (np.ndarray): Velocidad articular deseada [rad/s].
        ddq_des (np.ndarray): Aceleración articular deseada [rad/s²].
        kp      (np.ndarray): Ganancia proporcional (matriz diagonal).
        kv      (np.ndarray): Ganancia derivativa (matriz diagonal).
        F_ext   (np.ndarray): Fuerza externa en EF para feedback háptico [N].

    Returns:
        tuple: (tau, e, de) — torques [Nm], error posición, error velocidad.
    """
    # Error de seguimiento
    e  = q_des  - q
    de = dq_des - dq

    # Aceleración de referencia (control PD en espacio articular)
    a_d = ddq_des + kv @ de + kp @ e

    # Matrices dinámicas
    M_mat = inertia_matrix(q)
    C_mat = coriolis_matrix(q, dq)
    g_vec = gravity_vector(q)

    # Torque de control (cancela no-linealidades)
    tau = M_mat @ a_d + C_mat @ dq + g_vec

    # Agregar feedback háptico si hay fuerza externa medida
    if F_ext is not None and np.linalg.norm(F_ext) > 0.01:
        J = jacobian_3r(q)
        tau += J.T @ F_ext   # τ_fb = Jᵀ · β · Fₑ

    # Saturar torques para seguridad
    tau = np.clip(tau, -20.0, 20.0)
    return tau, e, de


# ──────────────────────────────────────────────
# INTEGRADOR EULER (dinámica del robot)
# ──────────────────────────────────────────────
def integrate_dynamics(q, dq, tau, dt=DT):
    """
    Integra la dinámica del robot usando Euler explícito.
    Ecuación: M(q)·q̈ = τ - C(q,q̇)·q̇ - g(q)

    Args:
        q   (np.ndarray): Posición articular actual.
        dq  (np.ndarray): Velocidad articular actual.
        tau (np.ndarray): Vector de torques aplicados.
        dt  (float):      Paso de integración [s].

    Returns:
        tuple: (q_new, dq_new) — nueva posición y velocidad articular.
    """
    M_mat = inertia_matrix(q)
    C_mat = coriolis_matrix(q, dq)
    g_vec = gravity_vector(q)

    # Aceleración articular resultante
    ddq = np.linalg.solve(M_mat, tau - C_mat @ dq - g_vec)

    # Integración Euler
    dq_new = dq + ddq * dt
    q_new  = q  + dq_new * dt

    # Aplicar límites articulares
    q_limits = np.array([np.pi/2, 2*np.pi/3, np.pi/2])
    q_new  = np.clip(q_new,  -q_limits, q_limits)
    dq_new = np.clip(dq_new, -3.0, 3.0)   # límite de velocidad [rad/s]
    return q_new, dq_new


# ──────────────────────────────────────────────
# COMUNICACIÓN DE RED — MAESTRO (CLIENTE)
# ──────────────────────────────────────────────
class MasterNetClient:
    """
    Cliente UDP del maestro. Envía posición cartesiana al esclavo
    y recibe fuerzas de contacto para feedback háptico.

    Protocolo JSON: {"xd": [x, y], "gripper": bool}
    Respuesta:      {"Fe": [Fx, Fy], "contact": bool}
    """
    def __init__(self, slave_ip, port_tx=9001, port_rx=9002):
        self.slave_ip = slave_ip
        self.port_tx  = port_tx
        self.port_rx  = port_rx
        self.sock_tx  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind(('', self.port_rx))
        self.sock_rx.settimeout(0.005)   # no bloquear el loop de control
        self.Fe       = np.zeros(2)      # fuerza recibida del esclavo [N]
        self.contact  = False
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def send_command(self, xd, gripper=True):
        """Envía posición deseada al esclavo."""
        msg = json.dumps({"xd": xd.tolist(), "gripper": int(gripper)})
        self.sock_tx.sendto(msg.encode(), (self.slave_ip, self.port_tx))

    def _recv_loop(self):
        """Hilo receptor — actualiza Fe y estado de contacto."""
        while True:
            try:
                data, _ = self.sock_rx.recvfrom(256)
                parsed  = json.loads(data.decode())
                self.Fe      = np.array(parsed["Fe"])
                self.contact = bool(parsed["contact"])
            except (socket.timeout, json.JSONDecodeError):
                pass


# ──────────────────────────────────────────────
# CLASE PRINCIPAL — ROBOT MAESTRO
# ──────────────────────────────────────────────
class MasterRobot:
    """
    Simulador completo del robot maestro con:
    - Dinámica 3R con Computed Torque
    - Control cartesiano por teclado
    - Feedback háptico desde el esclavo
    - Graficación en tiempo real (4 subplots)
    - Registro de datos para análisis
    """
    def __init__(self, slave_ip="127.0.0.1"):
        # Estado inicial del robot
        self.q  = np.array([0.4, -0.3, 0.2])    # posición articular [rad]
        self.dq = np.zeros(3)                     # velocidad articular
        self.q_des  = self.q.copy()
        self.dq_des = np.zeros(3)
        self.ddq_des = np.zeros(3)

        # Velocidad de movimiento cartesiano (controlado por teclado)
        self.v_cart = np.zeros(2)
        self.v_step = 0.5  # [m/s] por pulsación de tecla

        # Red
        self.net = MasterNetClient(slave_ip)

        # Histórico para graficación (buffer circular de 500 muestras)
        N = 500
        self.hist_t    = np.zeros(N)
        self.hist_q    = np.zeros((N, 3))
        self.hist_tau  = np.zeros((N, 3))
        self.hist_Fe   = np.zeros((N, 2))
        self.hist_x    = np.zeros((N, 2))
        self.idx = 0
        self.t   = 0.0

        # Estado de la pinza
        self.gripper_open = False

        # Referencia cinemática inversa (DLS)
        self.q_des = self.q.copy()

    def ik_dls(self, x_des, damp=0.01):
        """
        Cinemática inversa numérica por Damped Least Squares (DLS).
        Actualiza q_des para seguir la posición cartesiana deseada.

        Args:
            x_des (np.ndarray): Posición cartesiana deseada [m].
            damp  (float):      Factor de amortiguamiento (evita singularidades).
        """
        for _ in range(5):   # iteraciones internas
            x_cur = fk_3r(self.q_des)
            e_x   = x_des - x_cur
            if np.linalg.norm(e_x) < 1e-4:
                break
            J    = jacobian_3r(self.q_des)
            # Pseudoinversa DLS: J^+ = Jᵀ(JJᵀ + λ²I)^{-1}
            JJT  = J @ J.T
            Jp   = J.T @ np.linalg.inv(JJT + damp**2 * np.eye(2))
            self.q_des = self.q_des + Jp @ e_x

    def step(self):
        """
        Ejecuta un paso de simulación (dt = DT):
        1. Actualizar referencia cartesiana
        2. IK para obtener q_des
        3. Computed Torque → τ
        4. Integrar dinámica → q, q̇
        5. Enviar comando al esclavo
        6. Registrar datos
        """
        # 1. Referencia cartesiana = posición actual + v_cart*dt
        x_cur = fk_3r(self.q)
        x_des = x_cur + self.v_cart * DT

        # 2. Cinemática inversa
        self.ik_dls(x_des)

        # 3. Computed Torque con feedback háptico
        tau, e, de = computed_torque(
            self.q, self.dq, self.q_des, self.dq_des, self.ddq_des,
            F_ext = 0.3 * self.net.Fe  # factor β de feedback háptico
        )

        # 4. Integrar dinámica
        self.q, self.dq = integrate_dynamics(self.q, self.dq, tau)

        # 5. Enviar posición cartesiana al esclavo
        x_ef = fk_3r(self.q)
        self.net.send_command(x_ef, gripper=not self.gripper_open)

        # 6. Registrar datos para gráficas
        i = self.idx % 500
        self.hist_t[i]   = self.t
        self.hist_q[i]   = self.q
        self.hist_tau[i] = tau
        self.hist_Fe[i]  = self.net.Fe
        self.hist_x[i]   = x_ef
        self.idx += 1
        self.t   += DT


# ──────────────────────────────────────────────
# GRAFICACIÓN EN TIEMPO REAL
# ──────────────────────────────────────────────
def setup_plots(robot):
    """
    Configura la figura de graficación con 4 paneles:
    1. Vista 2D del robot (cinemática)
    2. Torques articulares τ₁, τ₂, τ₃ vs tiempo
    3. Fuerzas de contacto Fx, Fy vs tiempo
    4. Ángulos articulares q₁, q₂, q₃ vs tiempo
    """
    fig = plt.figure(figsize=(14, 10),
                     facecolor='#0a0a1a')
    fig.suptitle('TE3001B — Robot Maestro 3R | Peg-in-Hole Teleoperado',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    # Paleta de colores
    C = ['#00BFFF', '#FF6B6B', '#69FF47', '#FFD700', '#FF69B4', '#00FFD0']
    bg = '#0d1117'
    grid_c = '#1e2530'

    ax_robot = fig.add_subplot(2, 2, 1, facecolor=bg)
    ax_tau   = fig.add_subplot(2, 2, 2, facecolor=bg)
    ax_force = fig.add_subplot(2, 2, 3, facecolor=bg)
    ax_q     = fig.add_subplot(2, 2, 4, facecolor=bg)

    for ax in [ax_robot, ax_tau, ax_force, ax_q]:
        ax.tick_params(colors='#aaa')
        ax.xaxis.label.set_color('#aaa')
        ax.yaxis.label.set_color('#aaa')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        ax.grid(True, color=grid_c, linestyle='--', alpha=0.5)

    # Panel 1: Robot 2D
    ax_robot.set_xlim(-0.9, 0.9)
    ax_robot.set_ylim(-0.9, 0.9)
    ax_robot.set_aspect('equal')
    ax_robot.set_title('Vista Cinemática 3R', fontsize=11)
    ax_robot.set_xlabel('x [m]'); ax_robot.set_ylabel('y [m]')
    link_line, = ax_robot.plot([], [], 'o-', color=C[0],
                                linewidth=3, markersize=8, markerfacecolor=C[1])
    ef_dot,    = ax_robot.plot([], [], 's', color=C[2], markersize=12,
                                markerfacecolor=C[3], zorder=5)
    # Dibujar agujero destino (HOLE)
    hole_x, hole_y = 0.55, 0.10
    ax_robot.add_patch(plt.Circle((hole_x, hole_y), 0.025,
                                   color='#FFD700', alpha=0.6))
    ax_robot.plot(hole_x, hole_y, 'x', color='white', markersize=8, markeredgewidth=2)
    ax_robot.text(hole_x+0.03, hole_y+0.03, 'HOLE', color='#FFD700', fontsize=8)

    # Panel 2: Torques articulares
    ax_tau.set_title('Torques Articulares τ [Nm]', fontsize=11)
    ax_tau.set_xlabel('Tiempo [s]'); ax_tau.set_ylabel('τ [Nm]')
    lines_tau = [ax_tau.plot([], [], color=C[i], linewidth=1.5,
                              label=f'τ{i+1}')[0] for i in range(3)]
    ax_tau.legend(loc='upper right', fontsize=9,
                  facecolor='#1a1a2e', labelcolor='white')
    ax_tau.axhline(y=0, color='#444', linewidth=0.8)

    # Panel 3: Fuerzas de contacto
    ax_force.set_title('Fuerzas de Contacto Reflejadas [N]', fontsize=11)
    ax_force.set_xlabel('Tiempo [s]'); ax_force.set_ylabel('F [N]')
    line_Fx, = ax_force.plot([], [], color=C[4], linewidth=1.8, label='Fx')
    line_Fy, = ax_force.plot([], [], color=C[5], linewidth=1.8, label='Fy')
    ax_force.legend(loc='upper right', fontsize=9,
                    facecolor='#1a1a2e', labelcolor='white')
    ax_force.axhline(y=0, color='#444', linewidth=0.8)

    # Panel 4: Ángulos articulares
    ax_q.set_title('Ángulos Articulares q [rad]', fontsize=11)
    ax_q.set_xlabel('Tiempo [s]'); ax_q.set_ylabel('q [rad]')
    lines_q = [ax_q.plot([], [], color=C[i], linewidth=1.5,
                          label=f'q{i+1}', linestyle=['solid','dashed','dotted'][i])[0]
               for i in range(3)]
    ax_q.legend(loc='upper right', fontsize=9,
                facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    return fig, (ax_robot, ax_tau, ax_force, ax_q), \
           (link_line, ef_dot), lines_tau, (line_Fx, line_Fy), lines_q


def main(slave_ip):
    robot = MasterRobot(slave_ip)
    fig, axes, arm_lines, lines_tau, force_lines, lines_q = setup_plots(robot)
    ax_robot, ax_tau, ax_force, ax_q = axes
    link_line, ef_dot = arm_lines
    line_Fx, line_Fy  = force_lines

    # Hilo de simulación en background
    running = [True]
    def sim_loop():
        while running[0]:
            robot.step()
            time.sleep(DT)
    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    def animate(frame):
        n  = min(robot.idx, 500)
        i0 = robot.idx % 500
        # Reordenar buffer circular
        idx_range = np.arange(i0, i0 + n) % 500
        t   = robot.hist_t[idx_range]
        tau = robot.hist_tau[idx_range]
        Fe  = robot.hist_Fe[idx_range]
        q_h = robot.hist_q[idx_range]

        # Cinemática actual
        pts = fk_3r_full(robot.q)
        link_line.set_data(pts[:,0], pts[:,1])
        ef_dot.set_data([pts[-1,0]], [pts[-1,1]])

        # Ventana de tiempo visible (últimos 5 s)
        t_win = 5.0
        mask  = (t > robot.t - t_win) if robot.t > t_win else np.ones(n, bool)

        for i, ln in enumerate(lines_tau):
            ln.set_data(t[mask], tau[mask, i])
        ax_tau.set_xlim(max(0, robot.t - t_win), max(t_win, robot.t))
        ax_tau.relim(); ax_tau.autoscale_view(scalex=False)

        line_Fx.set_data(t[mask], Fe[mask, 0])
        line_Fy.set_data(t[mask], Fe[mask, 1])
        ax_force.set_xlim(max(0, robot.t - t_win), max(t_win, robot.t))
        ax_force.relim(); ax_force.autoscale_view(scalex=False)

        for i, ln in enumerate(lines_q):
            ln.set_data(t[mask], q_h[mask, i])
        ax_q.set_xlim(max(0, robot.t - t_win), max(t_win, robot.t))
        ax_q.relim(); ax_q.autoscale_view(scalex=False)

        return [link_line, ef_dot] + lines_tau + [line_Fx, line_Fy] + lines_q

    # Captura de teclas (movimiento cartesiano)
    def on_key_press(event):
        v = robot.v_step
        if   event.key == 'w': robot.v_cart = np.array([0, v])
        elif event.key == 's': robot.v_cart = np.array([0, -v])
        elif event.key == 'd': robot.v_cart = np.array([v, 0])
        elif event.key == 'a': robot.v_cart = np.array([-v, 0])
        elif event.key == 'q': robot.gripper_open = True
        elif event.key == 'e': robot.gripper_open = False
        elif event.key == 'escape':
            running[0] = False
            plt.close()

    def on_key_release(event):
        robot.v_cart = np.zeros(2)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    ani = animation.FuncAnimation(fig, animate, interval=50,   # 20 fps
                                  blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TE3001B — Robot Maestro Peg-in-Hole")
    parser.add_argument("--slave-ip", default="127.0.0.1",
                        help="IP del PC esclavo (default: loopback)")
    args = parser.parse_args()
    main(args.slave_ip)
