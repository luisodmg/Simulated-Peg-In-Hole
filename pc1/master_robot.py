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
    R    → RESET — regresa robot a posición inicial
    ESC  → salir
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
mpl.rcParams['keymap.save'] = []
mpl.rcParams['keymap.quit'] = []
mpl.rcParams['keymap.fullscreen'] = []
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket, threading, time, json, argparse

# ──────────────────────────────────────────────
# PARÁMETROS DEL ROBOT 3R
# ──────────────────────────────────────────────
L1, L2, L3 = 0.35, 0.30, 0.20
M1, M2, M3 = 1.5, 1.0, 0.5
G_GRAV      = 9.81

KP = np.diag([120.0, 100.0, 80.0])
KV = np.diag([25.0,  20.0,  15.0])

DT           = 0.02    # 50 Hz
HAPTIC_BETA  = 0.25    # escala del feedback háptico

Q_INIT = np.array([0.4, -0.3, 0.2])   # posición articular inicial


# ──────────────────────────────────────────────
# CINEMÁTICA
# ──────────────────────────────────────────────
def fk_3r(q):
    q1, q2, q3 = q
    x1 = L1*np.cos(q1);             y1 = L1*np.sin(q1)
    x2 = x1+L2*np.cos(q1+q2);      y2 = y1+L2*np.sin(q1+q2)
    x3 = x2+L3*np.cos(q1+q2+q3);   y3 = y2+L3*np.sin(q1+q2+q3)
    return np.array([x3, y3])

def fk_3r_full(q):
    q1, q2, q3 = q
    p0 = np.zeros(2)
    p1 = np.array([L1*np.cos(q1), L1*np.sin(q1)])
    p2 = p1+np.array([L2*np.cos(q1+q2), L2*np.sin(q1+q2)])
    p3 = p2+np.array([L3*np.cos(q1+q2+q3), L3*np.sin(q1+q2+q3)])
    return np.array([p0, p1, p2, p3])

def jacobian_3r(q):
    q1, q2, q3 = q
    s1=np.sin(q1); s12=np.sin(q1+q2); s123=np.sin(q1+q2+q3)
    c1=np.cos(q1); c12=np.cos(q1+q2); c123=np.cos(q1+q2+q3)
    return np.array([
        [-L1*s1-L2*s12-L3*s123, -L2*s12-L3*s123, -L3*s123],
        [ L1*c1+L2*c12+L3*c123,  L2*c12+L3*c123,  L3*c123]
    ])

def manipulability(q):
    J = jacobian_3r(q)
    return np.sqrt(max(0.0, np.linalg.det(J @ J.T)))


# ──────────────────────────────────────────────
# DINÁMICA — cada función llamada UNA SOLA VEZ por paso
# ──────────────────────────────────────────────
def inertia_matrix(q):
    q1, q2, q3 = q
    c2=np.cos(q2); c3=np.cos(q3); c23=np.cos(q2+q3)
    m11=(M1*L1**2+M2*(L1**2+L2**2+2*L1*L2*c2)+
         M3*(L1**2+L2**2+L3**2+2*L1*L2*c2+2*L1*L3*c23+2*L2*L3*c3))
    m12=M2*(L2**2+L1*L2*c2)+M3*(L2**2+L3**2+L1*L2*c2+L1*L3*c23+2*L2*L3*c3)
    m13=M3*(L3**2+L1*L3*c23+L2*L3*c3)
    m22=M2*L2**2+M3*(L2**2+L3**2+2*L2*L3*c3)
    m23=M3*(L3**2+L2*L3*c3)
    m33=M3*L3**2
    return np.array([[m11,m12,m13],[m12,m22,m23],[m13,m23,m33]])

def coriolis_matrix(q, dq):
    eps=1e-5; C=np.zeros((3,3))
    for k in range(3):
        qp=q.copy(); qp[k]+=eps
        qm=q.copy(); qm[k]-=eps
        C += 0.5*(inertia_matrix(qp)-inertia_matrix(qm))/(2*eps)*dq[k]
    return C

def gravity_vector(q):
    q1,q2,q3=q
    c1=np.cos(q1); c12=np.cos(q1+q2); c123=np.cos(q1+q2+q3)
    g1=G_GRAV*((M1+M2+M3)*L1*c1+(M2+M3)*L2*c12+M3*L3*c123)
    g2=G_GRAV*((M2+M3)*L2*c12+M3*L3*c123)
    g3=G_GRAV*M3*L3*c123
    return np.array([g1,g2,g3])


# ──────────────────────────────────────────────
# PASO DE SIMULACIÓN UNIFICADO
# Calcula M, C, g UNA SOLA VEZ y los reutiliza
# en el controlador y en el integrador.
# Antes: ~12 llamadas a inertia_matrix por paso
# Ahora: 7 llamadas a inertia_matrix por paso
# ──────────────────────────────────────────────
def sim_step(q, dq, q_des, dq_des, ddq_des, F_ext=None, dt=DT):
    """
    Ejecuta control + integración en un solo bloque compartiendo M, C, g.

    Returns:
        q_new, dq_new, tau, e, de
    """
    # ── Matrices dinámicas (calculadas UNA VEZ) ─────────────────────────
    M_mat = inertia_matrix(q)
    C_mat = coriolis_matrix(q, dq)
    g_vec = gravity_vector(q)

    # ── Computed Torque ──────────────────────────────────────────────────
    e   = q_des  - q
    de  = dq_des - dq
    a_d = ddq_des + KV@de + KP@e
    tau = M_mat@a_d + C_mat@dq + g_vec

    if F_ext is not None and np.linalg.norm(F_ext) > 0.01:
        J    = jacobian_3r(q)
        tau += J.T @ F_ext

    tau = np.clip(tau, -20.0, 20.0)

    # ── Integración Euler (reutiliza M, C, g) ───────────────────────────
    ddq    = np.linalg.solve(M_mat, tau - C_mat@dq - g_vec)
    dq_new = np.clip(dq + ddq*dt, -3.0, 3.0)
    q_lim  = np.array([np.pi/2, 2*np.pi/3, np.pi/2])
    q_new  = np.clip(q + dq_new*dt, -q_lim, q_lim)

    return q_new, dq_new, tau, e, de


# ──────────────────────────────────────────────
# RED
# ──────────────────────────────────────────────
class MasterNetClient:
    def __init__(self, slave_ip, port_tx=9001, port_rx=9002):
        self.slave_ip      = slave_ip
        self.sock_tx       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx       = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_rx.bind(('', port_rx))
        self.sock_rx.settimeout(0.005)
        self.Fe              = np.zeros(2)
        self.contact         = False
        self.reset_requested = False   # el esclavo pidió reset
        self._reset_count    = 0       # paquetes de reset pendientes de enviar
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def request_reset(self):
        """Llama esto al presionar R en el maestro: manda reset al esclavo."""
        self._reset_count = 10   # enviará el flag en los próximos 10 paquetes

    def send_command(self, xd, gripper=True):
        do_reset = self._reset_count > 0
        if do_reset:
            self._reset_count -= 1
        msg = json.dumps({
            "xd":      xd.tolist(),
            "gripper": int(gripper),
            "reset":   int(do_reset)   # ← flag de reset al esclavo
        })
        self.sock_tx.sendto(msg.encode(), (self.slave_ip, 9001))

    def _recv_loop(self):
        while True:
            try:
                data, _ = self.sock_rx.recvfrom(256)
                parsed  = json.loads(data.decode())
                self.Fe      = np.array(parsed["Fe"])
                self.contact = bool(parsed["contact"])
                # El esclavo puede pedir reset al maestro
                if parsed.get("reset", 0):
                    self.reset_requested = True
            except (socket.timeout, json.JSONDecodeError):
                pass


# ──────────────────────────────────────────────
# ROBOT MAESTRO
# ──────────────────────────────────────────────
class MasterRobot:
    def __init__(self, slave_ip="127.0.0.1"):
        self.net = MasterNetClient(slave_ip)
        self._reset_state()

        N = 500
        self.hist_t   = np.zeros(N)
        self.hist_q   = np.zeros((N, 3))
        self.hist_tau = np.zeros((N, 3))
        self.hist_Fe  = np.zeros((N, 2))
        self.hist_x   = np.zeros((N, 2))
        self.hist_w   = np.zeros(N)
        self.idx = 0
        self.t   = 0.0

        self.v_cart       = np.zeros(2)
        self.v_step       = 1.2
        self.gripper_open = False
        self.resetting    = False
        self._settle      = 0    # contador de estabilización post-reset

    def _reset_state(self):
        """Regresa el robot a la posición inicial. Llamado por tecla R."""
        self.q       = Q_INIT.copy()
        self.dq      = np.zeros(3)
        self.q_des   = Q_INIT.copy()
        self.dq_des  = np.zeros(3)
        self.ddq_des = np.zeros(3)

    def _do_reset(self):
        """Lógica de reset pura (sin propagar por red)."""
        self._reset_state()
        N = 500
        self.hist_t[:]   = 0.0
        self.hist_q[:]   = 0.0
        self.hist_tau[:] = 0.0
        self.hist_Fe[:]  = 0.0
        self.hist_x[:]   = 0.0
        self.hist_w[:]   = 0.0
        self.idx       = 0
        self.t         = 0.0
        self.resetting = True
        # Limpiar estado transitorio que causa el spike post-reset
        self.v_cart        = np.zeros(2)    # por si quedó una tecla presionada
        self.net.Fe        = np.zeros(2)    # fuerza antigua → torque spike
        self.net.contact   = False
        self._settle       = 50             # ignorar haptic 50 pasos (~1 s)

    def reset(self):
        """Reset iniciado por el usuario (tecla R): resetea y avisa al esclavo."""
        self._do_reset()
        self.net.request_reset()   # propaga al esclavo

    def ik_dls(self, x_des):
        """
        IK DLS + NULL-SPACE con zona muerta al 55%.
        Actúa más temprano (55% vs 70%) y más fuerte (K=0.6 vs 0.35)
        para que los joints nunca lleguen al 95%.
        """
        Q_LIM     = np.array([np.pi/2, 2*np.pi/3, np.pi/2])
        Q_MID     = np.array([0.0,     0.2,        0.0    ])
        K_NULL    = 0.6    # más fuerte para vencer al usuario cuando se acerca al límite
        THRESHOLD = 0.55   # actúa desde el 55% — antes de que sea tarde

        for _ in range(10):
            w     = manipulability(self.q_des)
            damp  = 0.01 if w > 0.05 else 0.15
            x_cur = fk_3r(self.q_des)
            e_x   = x_des - x_cur
            if np.linalg.norm(e_x) < 1e-4:
                break

            J   = jacobian_3r(self.q_des)
            Jp  = J.T @ np.linalg.inv(J@J.T + damp**2*np.eye(2))
            dq_primary = Jp @ e_x

            ratio   = np.abs(self.q_des) / Q_LIM
            # Ganancia escalonada: más fuerte cuanto más cerca del límite
            # 55-70%: K_NULL×0.5,  70-85%: K_NULL×1.0,  >85%: K_NULL×2.0
            scale   = np.where(ratio > 0.85, 2.0,
                      np.where(ratio > 0.70, 1.0,
                      np.where(ratio > 0.55, 0.5, 0.0)))
            grad_h  = scale * (self.q_des - Q_MID) / (Q_LIM**2)
            N_proj  = np.eye(3) - Jp @ J
            dq_null = N_proj @ (-K_NULL * grad_h)

            dq = np.clip(dq_primary + dq_null, -0.4, 0.4)
            self.q_des = np.clip(self.q_des + dq, -Q_LIM, Q_LIM)


    def step(self):
        # Si el esclavo pidió reset, ejecutarlo SIN propagarlo de vuelta
        if self.net.reset_requested:
            self.net.reset_requested = False
            self._do_reset()   # ← no llama request_reset(), corta el loop
            return

        # 1. Referencia cartesiana
        # Reducir velocidad si algún joint supera el 80% del límite
        Q_LIM  = np.array([np.pi/2, 2*np.pi/3, np.pi/2])
        ratio  = np.max(np.abs(self.q) / Q_LIM)
        v_scale = 1.0 if ratio < 0.80 else max(0.2, 1.0 - (ratio - 0.80) / 0.20)
        x_cur = fk_3r(self.q)
        x_des = x_cur + self.v_cart * DT * v_scale
        x_des[0] = np.clip(x_des[0], -0.7, 0.7)
        x_des[1] = np.clip(x_des[1], -0.5, 0.7)
        r = np.linalg.norm(x_des)
        r_max = L1+L2+L3-0.05
        if r > r_max:
            x_des = x_des/r*r_max

        # 2. IK
        self.ik_dls(x_des)

        # Período de estabilización post-reset: sin haptic, sin movimiento
        if self._settle > 0:
            self._settle -= 1
            F_haptic = np.zeros(2)
        else:
            F_haptic = HAPTIC_BETA * self.net.Fe

        # 3. Control + integración (M/C/g calculados UNA VEZ)
        self.q, self.dq, tau, e, de = sim_step(
            self.q, self.dq, self.q_des, self.dq_des, self.ddq_des,
            F_ext=F_haptic
        )

        # 4. Enviar al esclavo
        x_ef = fk_3r(self.q)
        self.net.send_command(x_ef, gripper=not self.gripper_open)

        # 5. Enviar telemetría al recorder (puerto 9003) — ignorar si no hay recorder
        try:
            telem = json.dumps({
                "src": "master",
                "t":   self.t,
                "q":   self.q.tolist(),
                "tau": tau.tolist(),
                "Fe":  self.net.Fe.tolist(),
                "x":   x_ef.tolist(),
                "w":   float(manipulability(self.q))
            })
            self.net.sock_tx.sendto(telem.encode(), ("127.0.0.1", 9003))
        except Exception:
            pass

        # 6. Registrar
        i = self.idx % 500
        self.hist_t[i]   = self.t
        self.hist_q[i]   = self.q
        self.hist_tau[i] = tau
        self.hist_Fe[i]  = self.net.Fe
        self.hist_x[i]   = x_ef
        self.hist_w[i]   = manipulability(self.q)
        self.idx += 1
        self.t   += DT


# ──────────────────────────────────────────────
# GRAFICACIÓN
# ──────────────────────────────────────────────
def setup_plots(robot):
    fig = plt.figure(figsize=(14, 10), facecolor='#0a0a1a')
    fig.suptitle('TE3001B — Robot Maestro 3R | Peg-in-Hole Teleoperado   [R]=Reset',
                 color='white', fontsize=13, fontweight='bold', y=0.98)

    C      = ['#00BFFF','#FF6B6B','#69FF47','#FFD700','#FF69B4','#00FFD0']
    bg     = '#0d1117'
    grid_c = '#1e2530'

    ax_robot = fig.add_subplot(2,2,1, facecolor=bg)
    ax_tau   = fig.add_subplot(2,2,2, facecolor=bg)
    ax_force = fig.add_subplot(2,2,3, facecolor=bg)
    ax_q     = fig.add_subplot(2,2,4, facecolor=bg)

    for ax in [ax_robot, ax_tau, ax_force, ax_q]:
        ax.tick_params(colors='#aaa')
        ax.xaxis.label.set_color('#aaa')
        ax.yaxis.label.set_color('#aaa')
        ax.title.set_color('white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.grid(True, color=grid_c, linestyle='--', alpha=0.5)

    # Panel 1: Robot 2D
    ax_robot.set_xlim(-0.9, 0.9); ax_robot.set_ylim(-0.9, 0.9)
    ax_robot.set_aspect('equal')
    ax_robot.set_title('Vista Cinemática 3R', fontsize=11)
    ax_robot.set_xlabel('x [m]'); ax_robot.set_ylabel('y [m]')
    link_line, = ax_robot.plot([], [], 'o-', color=C[0], linewidth=3,
                                markersize=8, markerfacecolor=C[1])
    ef_dot,    = ax_robot.plot([], [], 's',  color=C[2], markersize=12,
                                markerfacecolor=C[3], zorder=5)
    ax_robot.add_patch(plt.Circle((0.55, 0.20), 0.025, color='#FFD700', alpha=0.6))
    ax_robot.plot(0.55, 0.20, 'x', color='white', markersize=8, markeredgewidth=2)
    ax_robot.text(0.58, 0.23, 'HOLE', color='#FFD700', fontsize=8)

    reset_text = ax_robot.text(0.5, 0.5, '', transform=ax_robot.transAxes,
                                color='#00FF88', fontsize=16, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='#0a0a1a',
                                          alpha=0.85, edgecolor='#00FF88'))
    # Aviso de límite articular (aparece cuando null-space está activo)
    limit_text = ax_robot.text(0.02, 0.04, '', transform=ax_robot.transAxes,
                                color='#FF8800', fontsize=9, fontweight='bold',
                                verticalalignment='bottom')

    # Panel 2: Torques
    ax_tau.set_title('Torques Articulares τ [Nm]', fontsize=11)
    ax_tau.set_xlabel('Tiempo [s]'); ax_tau.set_ylabel('τ [Nm]')
    lines_tau = [ax_tau.plot([], [], color=C[i], linewidth=1.5,
                              label=f'τ{i+1}')[0] for i in range(3)]
    ax_tau.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    ax_tau.axhline(y=0, color='#444', linewidth=0.8)

    # Panel 3: Fuerzas reflejadas (haptic feedback)
    ax_force.set_title('★ Fuerzas Reflejadas [N]  ← Haptic Feedback', fontsize=10)
    ax_force.set_xlabel('Tiempo [s]'); ax_force.set_ylabel('F [N]')
    line_Fx, = ax_force.plot([], [], color=C[4], linewidth=1.8, label='Fx')
    line_Fy, = ax_force.plot([], [], color=C[5], linewidth=1.8, label='Fy')
    ax_force.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    ax_force.axhline(y=0, color='#444', linewidth=0.8)
    ax_force.text(0.02, 0.04, f'β={HAPTIC_BETA} → τ=Jᵀ·β·Fe',
                  transform=ax_force.transAxes, color='#aaa', fontsize=8)

    # Panel 4: Ángulos + manipulabilidad
    ax_q.set_title('Ángulos Articulares q [rad]', fontsize=11)
    ax_q.set_xlabel('Tiempo [s]'); ax_q.set_ylabel('q [rad]')
    lines_q = [ax_q.plot([], [], color=C[i], linewidth=1.5,
                          label=f'q{i+1}',
                          linestyle=['solid','dashed','dotted'][i])[0]
               for i in range(3)]
    ax_q.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    ax_w = ax_q.twinx()
    ax_w.set_ylabel('w [manipulabilidad]', color='#FF4444')
    ax_w.tick_params(colors='#FF4444')
    ax_w.axhline(y=0.05, color='#FF4444', linewidth=1.0, linestyle=':', alpha=0.7)
    line_w, = ax_w.plot([], [], color='#FF4444', linewidth=1.5,
                         linestyle='--', alpha=0.8, label='w')

    return (fig,
            (ax_robot, ax_tau, ax_force, ax_q),
            (link_line, ef_dot, reset_text, limit_text),
            lines_tau,
            (line_Fx, line_Fy),
            lines_q,
            (line_w, ax_w))


def main(slave_ip):
    robot = MasterRobot(slave_ip)
    (fig, axes, arm_artists, lines_tau,
     force_lines, lines_q, w_artists) = setup_plots(robot)

    ax_robot, ax_tau, ax_force, ax_q        = axes
    link_line, ef_dot, reset_text, limit_text = arm_artists
    line_Fx, line_Fy                         = force_lines
    line_w, ax_w                             = w_artists

    running      = [True]
    reset_frames = [0]   # contador para mostrar texto de reset

    # ── Sim loop con perf_counter (más preciso que time.sleep solo) ──────
    def sim_loop():
        while running[0]:
            t0 = time.perf_counter()
            try:
                robot.step()
            except Exception as ex:
                print(f"[sim_loop] error: {ex} — haciendo reset automático")
                robot.reset()
            elapsed = time.perf_counter() - t0
            remaining = DT - elapsed
            if remaining > 0:
                time.sleep(remaining)
            # Si el paso tardó más de DT no dormimos: el loop se mantiene
            # lo más cerca posible a 50 Hz sin bloquearse

    threading.Thread(target=sim_loop, daemon=True).start()

    def animate(frame):
        n  = min(robot.idx, 500)
        if n == 0:
            return [link_line, ef_dot]
        i0        = robot.idx % 500
        idx_range = np.arange(i0, i0+n) % 500
        t   = robot.hist_t[idx_range]
        tau = robot.hist_tau[idx_range]
        Fe  = robot.hist_Fe[idx_range]
        q_h = robot.hist_q[idx_range]
        w_h = robot.hist_w[idx_range]

        pts = fk_3r_full(robot.q)
        link_line.set_data(pts[:,0], pts[:,1])
        ef_dot.set_data([pts[-1,0]], [pts[-1,1]])

        # Texto de reset (aparece 1 segundo ≈ 20 frames a 20fps)
        if robot.resetting:
            reset_text.set_text('RESET ✓')
            reset_frames[0] = 20
            robot.resetting = False
        if reset_frames[0] > 0:
            reset_text.set_text('RESET ✓')
            reset_frames[0] -= 1
        else:
            reset_text.set_text('')

        t_win = 5.0
        mask  = (t > robot.t - t_win) if robot.t > t_win else np.ones(n, bool)

        for i, ln in enumerate(lines_tau):
            ln.set_data(t[mask], tau[mask, i])
        ax_tau.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        ax_tau.relim(); ax_tau.autoscale_view(scalex=False)

        line_Fx.set_data(t[mask], Fe[mask,0])
        line_Fy.set_data(t[mask], Fe[mask,1])
        ax_force.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        ax_force.relim(); ax_force.autoscale_view(scalex=False)

        for i, ln in enumerate(lines_q):
            ln.set_data(t[mask], q_h[mask, i])
        ax_q.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        ax_q.relim(); ax_q.autoscale_view(scalex=False)

        line_w.set_data(t[mask], w_h[mask])
        ax_w.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        ax_w.relim(); ax_w.autoscale_view(scalex=False)

        # Aviso de límite articular — se activa cuando algún joint supera 80% del límite
        Q_LIM   = np.array([np.pi/2, 2*np.pi/3, np.pi/2])
        q_ratio = np.abs(robot.q) / Q_LIM   # 0..1, donde 1 = límite
        near    = np.where(q_ratio > 0.80)[0]
        if len(near):
            nombres = ['q1','q2','q3']
            msg = '⚠ null-space activo: ' + ', '.join(
                f'{nombres[i]}={q_ratio[i]*100:.0f}%' for i in near)
            limit_text.set_text(msg)
        else:
            limit_text.set_text('')

        return ([link_line, ef_dot, reset_text, limit_text] + lines_tau +
                [line_Fx, line_Fy] + lines_q + [line_w])

    def on_key_press(event):
        v = robot.v_step
        if   event.key == 'w':      robot.v_cart = np.array([0.0,  v])
        elif event.key == 's':      robot.v_cart = np.array([0.0, -v])
        elif event.key == 'd':      robot.v_cart = np.array([ v, 0.0])
        elif event.key == 'a':      robot.v_cart = np.array([-v, 0.0])
        elif event.key == 'q':      robot.gripper_open = True
        elif event.key == 'e':      robot.gripper_open = False
        elif event.key == 'r':      robot.reset()        # ← RESET
        elif event.key == 'escape':
            running[0] = False
            plt.close()

    def on_key_release(event):
        robot.v_cart = np.zeros(2)

    fig.canvas.mpl_connect('key_press_event',   on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    ani = animation.FuncAnimation(fig, animate, interval=50,
                                  blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TE3001B — Robot Maestro")
    parser.add_argument("--slave-ip", default="127.0.0.1")
    args = parser.parse_args()
    main(args.slave_ip)