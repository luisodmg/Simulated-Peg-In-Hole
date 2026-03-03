"""
╔══════════════════════════════════════════════════════════════╗
║  ROBOT ESCLAVO — TE3001B  Peg-in-Hole Teleoperado           ║
║  Control de Impedancia + Detección de Contacto              ║
║  Graficación de fuerzas de contacto y torques               ║
╚══════════════════════════════════════════════════════════════╝

Ejecutar en la PC ESCLAVO:
    python3 slave_robot.py

Nodos de red:
    Escucha en UDP:9001 (recibe xd del maestro)
    Envía  en UDP:9002 al maestro (Fe, estado)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket, threading, time, json, argparse

# ─────────── parámetros del robot (idénticos al maestro) ─────────────────────
L1, L2, L3 = 0.35, 0.30, 0.20
M1, M2, M3 = 1.5, 1.0, 0.5
G_GRAV      = 9.81
DT          = 0.01

# ─────────── ganancias de impedancia ─────────────────────────────────────────
KD_IMP = 400.0   # rigidez virtual [N/m]    — Kd de la impedancia
BD_IMP = 40.0    # amortiguamiento virtual [N·s/m]
KP_ART = np.diag([100.0, 80.0, 60.0])   # ganancias Computed Torque
KV_ART = np.diag([20.0,  16.0, 12.0])

# ─────────── geometría del Peg-in-Hole ───────────────────────────────────────
PEG_LENGTH  = 0.08   # longitud del peg [m]
PEG_RADIUS  = 0.008  # radio del peg  [m]
HOLE_CENTER = np.array([0.55, 0.10])   # posición del agujero en espacio global
HOLE_RADIUS = 0.009  # radio del agujero (tolerancia 1mm)
F_CONTACT_K = 2000.0  # rigidez de contacto pared del agujero [N/m]
F_THRESHOLD = 2.0     # umbral de detección de contacto [N]


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES CINEMÁTICAS (reutilizadas del maestro)
# ──────────────────────────────────────────────────────────────────────────────
def fk_3r(q):
    q1, q2, q3 = q
    x1 = L1*np.cos(q1); y1 = L1*np.sin(q1)
    x2 = x1 + L2*np.cos(q1+q2); y2 = y1 + L2*np.sin(q1+q2)
    x3 = x2 + L3*np.cos(q1+q2+q3); y3 = y2 + L3*np.sin(q1+q2+q3)
    return np.array([x3, y3])

def fk_3r_full(q):
    q1, q2, q3 = q
    p0 = np.array([0.0, 0.0])
    p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
    p2 = p1 + np.array([L2*np.cos(q1+q2), L2*np.sin(q1+q2)])
    p3 = p2 + np.array([L3*np.cos(q1+q2+q3), L3*np.sin(q1+q2+q3)])
    return np.array([p0, p1, p2, p3])

def jacobian_3r(q):
    q1, q2, q3 = q
    s1   = np.sin(q1); s12  = np.sin(q1+q2); s123 = np.sin(q1+q2+q3)
    c1   = np.cos(q1); c12  = np.cos(q1+q2); c123 = np.cos(q1+q2+q3)
    return np.array([
        [-L1*s1-L2*s12-L3*s123, -L2*s12-L3*s123, -L3*s123],
        [ L1*c1+L2*c12+L3*c123,  L2*c12+L3*c123,  L3*c123]
    ])

def inertia_matrix(q):
    q1, q2, q3 = q
    c2 = np.cos(q2); c3 = np.cos(q3); c23 = np.cos(q2+q3)
    m11 = (M1*L1**2 + M2*(L1**2+L2**2+2*L1*L2*c2) +
           M3*(L1**2+L2**2+L3**2+2*L1*L2*c2+2*L1*L3*c23+2*L2*L3*c3))
    m12 = M2*(L2**2+L1*L2*c2) + M3*(L2**2+L3**2+L1*L2*c2+L1*L3*c23+2*L2*L3*c3)
    m13 = M3*(L3**2+L1*L3*c23+L2*L3*c3)
    m22 = M2*L2**2 + M3*(L2**2+L3**2+2*L2*L3*c3)
    m23 = M3*(L3**2+L2*L3*c3)
    m33 = M3*L3**2
    return np.array([[m11,m12,m13],[m12,m22,m23],[m13,m23,m33]])

def coriolis_matrix(q, dq):
    eps = 1e-5
    C = np.zeros((3,3))
    for k in range(3):
        qp = q.copy(); qp[k] += eps
        qm = q.copy(); qm[k] -= eps
        dM = (inertia_matrix(qp) - inertia_matrix(qm)) / (2*eps)
        C += 0.5 * dM * dq[k]
    return C

def gravity_vector(q):
    q1, q2, q3 = q
    c1 = np.cos(q1); c12 = np.cos(q1+q2); c123 = np.cos(q1+q2+q3)
    g1 = G_GRAV*((M1+M2+M3)*L1*c1 + (M2+M3)*L2*c12 + M3*L3*c123)
    g2 = G_GRAV*((M2+M3)*L2*c12 + M3*L3*c123)
    g3 = G_GRAV*M3*L3*c123
    return np.array([g1, g2, g3])

def integrate_dynamics(q, dq, tau, dt=DT):
    M_mat = inertia_matrix(q)
    C_mat = coriolis_matrix(q, dq)
    g_vec = gravity_vector(q)
    ddq = np.linalg.solve(M_mat, tau - C_mat@dq - g_vec)
    dq_new = np.clip(dq + ddq*dt, -3.0, 3.0)
    q_lim  = np.array([np.pi/2, 2*np.pi/3, np.pi/2])
    q_new  = np.clip(q + dq_new*dt, -q_lim, q_lim)
    return q_new, dq_new


# ──────────────────────────────────────────────────────────────────────────────
# MODELO DE CONTACTO PEG-IN-HOLE
# ──────────────────────────────────────────────────────────────────────────────
class PegHoleContact:
    """
    Modelo de contacto elástico para la inserción Peg-in-Hole.

    El peg está adherido al efector final. Se detecta contacto cuando
    el peg intenta entrar al agujero con desalineación mayor al umbral.
    La fuerza de contacto es proporcional a la penetración en la pared.

    Fases de la tarea:
        APPROACH  → el peg se aproxima al agujero desde arriba
        CONTACT   → el peg toca el borde del agujero (Fz > umbral)
        INSERTION → control mixto: Fx,Fy→0 (centrar) + z↓ (insertar)
        COMPLETE  → inserción exitosa
    """

    APPROACH, CONTACT, INSERTION, COMPLETE = 0, 1, 2, 3
    STATE_NAMES = {0: "APROXIMACIÓN", 1: "CONTACTO",
                   2: "INSERCIÓN", 3: "COMPLETADO ✓"}

    def __init__(self):
        self.phase    = self.APPROACH
        self.depth    = 0.0       # profundidad de inserción [m]

    def compute_contact_force(self, x_ef):
        """
        Calcula la fuerza de contacto ejercida sobre el peg.

        Modelo: resorte lineal con la pared del agujero.
        Si el peg está dentro del radio del agujero → sin contacto lateral.
        Si está fuera → fuerza proporcional a la penetración.

        Args:
            x_ef (np.ndarray): Posición del efector final [x, y] en metros.

        Returns:
            tuple: (F_contact [N], state_str, in_contact bool)
        """
        delta = x_ef - HOLE_CENTER          # vector EF → centro agujero
        dist  = np.linalg.norm(delta)       # distancia al centro

        F_contact = np.zeros(2)
        in_contact = False

        # Proximidad vertical: el peg está cerca del agujero
        if abs(x_ef[1] - HOLE_CENTER[1]) < PEG_LENGTH * 1.5:

            if dist < HOLE_RADIUS:
                # EF dentro del agujero — sin fuerza lateral, solo inserción
                self.phase    = max(self.phase, self.INSERTION)
                self.depth    = HOLE_CENTER[1] - x_ef[1]  # profundidad
                if self.depth > PEG_LENGTH * 0.85:
                    self.phase = self.COMPLETE
            elif dist < HOLE_RADIUS + 0.02:
                # Contacto con la pared del agujero — fuerza de reacción
                penetration   = dist - HOLE_RADIUS
                F_mag         = F_CONTACT_K * penetration
                F_contact     = -F_mag * (delta / dist)  # hacia el centro
                in_contact    = True
                if self.phase < self.CONTACT:
                    self.phase = self.CONTACT

        return F_contact, self.STATE_NAMES[self.phase], in_contact


# ──────────────────────────────────────────────────────────────────────────────
# CONTROL DE IMPEDANCIA EN ESPACIO DE TAREA
# ──────────────────────────────────────────────────────────────────────────────
def impedance_control(q, dq, x_des, dx_des, q_des_prev,
                      F_contact=None, kd=KD_IMP, bd=BD_IMP):
    """
    Control de impedancia para el esclavo en contacto.

    Emula el comportamiento masa-resorte-amortiguador:
        Md·ë_x + Bd·ė_x + Kd·e_x = F_ext

    Los torques se calculan vía Jacobiano transpuesto:
        τ_imp = Jᵀ(Kd·e_x + Bd·ė_x)

    Sobre estos torques se aplica compensación de gravedad
    y Coriolis para cancelar no-linealidades.

    Args:
        q, dq       : Estado articular actual.
        x_des       : Posición cartesiana deseada (del maestro).
        dx_des      : Velocidad cartesiana deseada.
        q_des_prev  : q_des del paso anterior (para derivada numérica).
        F_contact   : Fuerza de contacto medida [N].
        kd, bd      : Rigidez y amortiguamiento virtuales.

    Returns:
        tuple: (tau, F_total, e_x) — torques, fuerza resultante, error cartesiano.
    """
    x_cur  = fk_3r(q)
    J      = jacobian_3r(q)

    # Velocidad cartesiana actual
    dx_cur = J @ dq

    # Error cartesiano
    e_x  = x_des - x_cur
    de_x = dx_des - dx_cur

    # Fuerza de impedancia (ley de resorte-amortiguador)
    F_imp = kd * e_x + bd * de_x

    # Incluir fuerza de contacto si la hay
    F_total = F_imp.copy()
    if F_contact is not None:
        F_total += F_contact

    # Torques de impedancia: τ = Jᵀ · F_total
    tau_imp = J.T @ F_total

    # Compensación de gravedad (siempre necesaria)
    g_vec = gravity_vector(q)
    C_mat = coriolis_matrix(q, dq)
    tau   = tau_imp + g_vec + C_mat @ dq

    # Saturar torques
    tau = np.clip(tau, -20.0, 20.0)
    return tau, F_total, e_x


# ──────────────────────────────────────────────────────────────────────────────
# SERVIDOR DE RED — ESCLAVO
# ──────────────────────────────────────────────────────────────────────────────
class SlaveNetServer:
    """
    Servidor UDP del esclavo.
    Recibe comandos xd del maestro y envía fuerzas de contacto de vuelta.
    """
    def __init__(self, master_ip="127.0.0.1", port_rx=9001, port_tx=9002):
        self.master_ip = master_ip
        self.port_rx   = port_rx
        self.port_tx   = port_tx
        self.sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.port_rx))
        self.sock.settimeout(0.005)
        self.x_des    = np.array([0.55, 0.40])  # posición inicial deseada
        self.gripper  = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        """Hilo receptor — actualiza xd continuamente."""
        while True:
            try:
                data, addr = self.sock.recvfrom(256)
                parsed = json.loads(data.decode())
                self.x_des   = np.array(parsed["xd"])
                self.gripper = bool(parsed["gripper"])
                self.master_addr = addr   # guardar para responder
            except (socket.timeout, json.JSONDecodeError, AttributeError):
                pass

    def send_force(self, Fe, contact, master_port=9002):
        """Envía fuerza de contacto al maestro."""
        msg = json.dumps({"Fe": Fe.tolist(), "contact": int(contact)})
        try:
            self.sock.sendto(msg.encode(), (self.master_ip, master_port))
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL — ROBOT ESCLAVO
# ──────────────────────────────────────────────────────────────────────────────
class SlaveRobot:
    """
    Simulador completo del robot esclavo:
    - Dinámica 3R con control de impedancia
    - Modelo de contacto Peg-in-Hole
    - Graficación: fuerzas de contacto, torques, trayectoria EF
    - Comunicación bidireccional con el maestro
    """
    def __init__(self, master_ip="127.0.0.1"):
        # Estado inicial: robot extendido hacia el agujero
        self.q  = np.array([0.6, -0.5, 0.1])
        self.dq = np.zeros(3)
        self.q_des_prev = self.q.copy()

        # Modelos
        self.contact_model = PegHoleContact()
        self.net = SlaveNetServer(master_ip)

        # Histórico (buffer circular)
        N = 500
        self.hist_t      = np.zeros(N)
        self.hist_tau    = np.zeros((N, 3))
        self.hist_Fc     = np.zeros((N, 2))
        self.hist_x      = np.zeros((N, 2))
        self.hist_ex     = np.zeros((N, 2))
        self.idx = 0
        self.t   = 0.0
        self.contact_state = "APROXIMACIÓN"

    def ik_dls(self, x_des, damp=0.01):
        """Cinemática inversa por DLS — igual que en el maestro."""
        q = self.q.copy()
        for _ in range(8):
            e = x_des - fk_3r(q)
            if np.linalg.norm(e) < 1e-4:
                break
            J  = jacobian_3r(q)
            Jp = J.T @ np.linalg.inv(J @ J.T + damp**2 * np.eye(2))
            q  = q + Jp @ e
        return q

    def step(self):
        """Un paso de simulación del esclavo."""
        x_des = self.net.x_des.copy()

        # Calcular fuerza de contacto según modelo Peg-in-Hole
        x_ef = fk_3r(self.q)
        F_contact, state_str, in_contact = \
            self.contact_model.compute_contact_force(x_ef)

        self.contact_state = state_str

        # Velocidad cartesiana deseada (diferenciación numérica)
        dx_des = (x_des - fk_3r(self.ik_dls(x_des))) / DT * 0.0  # se usa 0

        # Control de impedancia con contacto
        tau, F_total, e_x = impedance_control(
            self.q, self.dq, x_des, dx_des,
            self.q_des_prev, F_contact
        )

        # Integrar dinámica
        self.q, self.dq = integrate_dynamics(self.q, self.dq, tau)

        # Enviar fuerza de contacto al maestro (para feedback háptico)
        self.net.send_force(F_contact, in_contact)

        # Registrar datos
        i = self.idx % 500
        self.hist_t[i]   = self.t
        self.hist_tau[i] = tau
        self.hist_Fc[i]  = F_contact
        self.hist_x[i]   = x_ef
        self.hist_ex[i]  = e_x
        self.idx += 1
        self.t   += DT


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICACIÓN ESCLAVO
# ──────────────────────────────────────────────────────────────────────────────
def setup_slave_plots(robot):
    """
    Configura 4 paneles para el esclavo:
    1. Vista 2D del robot + peg + agujero
    2. Fuerzas de contacto Fx, Fy vs tiempo
    3. Torques articulares
    4. Error cartesiano |e_x|, |e_y|
    """
    fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0f')
    fig.suptitle('TE3001B — Robot Esclavo 3R | Control de Impedancia + Peg-in-Hole',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    bg = '#0d1117'
    C  = ['#FF6B6B', '#69FF47', '#00BFFF', '#FFD700', '#FF69B4', '#00FFD0']

    ax_robot = fig.add_subplot(2, 2, 1, facecolor=bg)
    ax_force = fig.add_subplot(2, 2, 2, facecolor=bg)
    ax_tau   = fig.add_subplot(2, 2, 3, facecolor=bg)
    ax_err   = fig.add_subplot(2, 2, 4, facecolor=bg)

    for ax in [ax_robot, ax_force, ax_tau, ax_err]:
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        ax.grid(True, color='#1e2530', linestyle='--', alpha=0.5)
        ax.title.set_color('white')
        ax.xaxis.label.set_color('#aaa'); ax.yaxis.label.set_color('#aaa')

    # Panel 1: Robot + escenario
    ax_robot.set_xlim(-0.9, 0.9); ax_robot.set_ylim(-0.5, 1.0)
    ax_robot.set_aspect('equal')
    ax_robot.set_title('Esclavo — Peg-in-Hole', fontsize=11)
    ax_robot.set_xlabel('x [m]'); ax_robot.set_ylabel('y [m]')
    # Dibujar agujero
    hole_patch = plt.Rectangle(
        (HOLE_CENTER[0]-0.04, HOLE_CENTER[1]-0.005), 0.08, 0.04,
        color='#2a2a4a', zorder=1)
    ax_robot.add_patch(hole_patch)
    ax_robot.plot(*HOLE_CENTER, 'x', color='#FFD700', markersize=10,
                  markeredgewidth=2, zorder=3)
    ax_robot.text(HOLE_CENTER[0]+0.02, HOLE_CENTER[1]+0.015,
                  'HOLE', color='#FFD700', fontsize=8)

    link_line, = ax_robot.plot([], [], 'o-', color=C[2], linewidth=3,
                                markersize=8, markerfacecolor=C[0])
    peg_line,  = ax_robot.plot([], [], '-', color=C[3], linewidth=5, zorder=4)
    state_text  = ax_robot.text(0.02, 0.96, '', transform=ax_robot.transAxes,
                                 color='#FFD700', fontsize=10, fontweight='bold',
                                 verticalalignment='top')
    force_arrow = ax_robot.annotate('', xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='->', color='red',
                                                     lw=2.5), zorder=5)

    # Panel 2: Fuerzas de contacto
    ax_force.set_title('Fuerzas de Contacto [N]', fontsize=11)
    ax_force.set_xlabel('Tiempo [s]'); ax_force.set_ylabel('F [N]')
    line_Fx, = ax_force.plot([], [], color=C[4], linewidth=2.0, label='Fx contacto')
    line_Fy, = ax_force.plot([], [], color=C[5], linewidth=2.0, label='Fy contacto')
    ax_force.axhline(y=0, color='#444', linewidth=0.8)
    ax_force.axhline(y=F_THRESHOLD, color='#FF4444', linewidth=1.2,
                     linestyle='--', label=f'Umbral {F_THRESHOLD} N')
    ax_force.legend(loc='upper right', fontsize=8,
                    facecolor='#1a1a2e', labelcolor='white')

    # Panel 3: Torques
    ax_tau.set_title('Torques Articulares τ [Nm]', fontsize=11)
    ax_tau.set_xlabel('Tiempo [s]'); ax_tau.set_ylabel('τ [Nm]')
    lines_tau = [ax_tau.plot([], [], color=C[i], linewidth=1.5,
                              label=f'τ{i+1}')[0] for i in range(3)]
    ax_tau.axhline(y=0, color='#444', linewidth=0.8)
    ax_tau.legend(loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', labelcolor='white')

    # Panel 4: Error cartesiano
    ax_err.set_title('Error Cartesiano |e| [mm]', fontsize=11)
    ax_err.set_xlabel('Tiempo [s]'); ax_err.set_ylabel('Error [mm]')
    line_ex, = ax_err.plot([], [], color=C[0], linewidth=1.8, label='|eₓ|')
    line_ey, = ax_err.plot([], [], color=C[1], linewidth=1.8, label='|e_y|')
    ax_err.axhline(y=1.0, color='#888', linewidth=1.0,
                   linestyle=':', label='1 mm (meta)')
    ax_err.legend(loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return (fig,
            (ax_robot, ax_force, ax_tau, ax_err),
            (link_line, peg_line, state_text, force_arrow),
            (line_Fx, line_Fy),
            lines_tau,
            (line_ex, line_ey))


def main(master_ip):
    robot = SlaveRobot(master_ip)
    (fig, axes, robot_artists, force_lines, lines_tau, err_lines) = \
        setup_slave_plots(robot)
    ax_robot, ax_force, ax_tau, ax_err = axes
    link_line, peg_line, state_text, force_arrow = robot_artists
    line_Fx, line_Fy = force_lines
    line_ex, line_ey = err_lines

    running = [True]
    def sim_loop():
        while running[0]:
            robot.step()
            time.sleep(DT)
    threading.Thread(target=sim_loop, daemon=True).start()

    def animate(frame):
        n   = min(robot.idx, 500)
        i0  = robot.idx % 500
        idx = np.arange(i0, i0+n) % 500
        t   = robot.hist_t[idx]
        tau = robot.hist_tau[idx]
        Fc  = robot.hist_Fc[idx]
        ex  = robot.hist_ex[idx]
        t_win = 5.0
        mask  = (t > robot.t - t_win)

        # Robot
        pts = fk_3r_full(robot.q)
        link_line.set_data(pts[:,0], pts[:,1])
        # Peg visual
        ef = pts[-1]
        peg_dir = pts[-1] - pts[-2]
        if np.linalg.norm(peg_dir) > 0:
            peg_dir = peg_dir / np.linalg.norm(peg_dir)
        peg_start = ef
        peg_end   = ef + peg_dir * PEG_LENGTH
        peg_line.set_data([peg_start[0], peg_end[0]],
                          [peg_start[1], peg_end[1]])
        state_text.set_text(f"Estado: {robot.contact_state}")

        # Fuerzas
        ax_force.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        line_Fx.set_data(t[mask], Fc[mask, 0])
        line_Fy.set_data(t[mask], Fc[mask, 1])
        ax_force.relim(); ax_force.autoscale_view(scalex=False)

        # Torques
        ax_tau.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        for i, ln in enumerate(lines_tau):
            ln.set_data(t[mask], tau[mask, i])
        ax_tau.relim(); ax_tau.autoscale_view(scalex=False)

        # Error (en mm)
        ax_err.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        line_ex.set_data(t[mask], np.abs(ex[mask, 0])*1000)
        line_ey.set_data(t[mask], np.abs(ex[mask, 1])*1000)
        ax_err.relim(); ax_err.autoscale_view(scalex=False)

        return ([link_line, peg_line] + lines_tau + [line_Fx, line_Fy,
                                                      line_ex, line_ey])

    ani = animation.FuncAnimation(fig, animate, interval=50,
                                  blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TE3001B — Robot Esclavo")
    parser.add_argument("--master-ip", default="127.0.0.1",
                        help="IP del PC maestro (default: loopback)")
    args = parser.parse_args()
    main(args.master_ip)
