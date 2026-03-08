"""
╔══════════════════════════════════════════════════════════════╗
║  ROBOT ESCLAVO — TE3001B  Peg-in-Hole Teleoperado           ║
║  Control de Impedancia + Detección de Contacto              ║
║  Graficación de fuerzas de contacto y torques               ║
╚══════════════════════════════════════════════════════════════╝

Ejecutar en la PC ESCLAVO:
    python3 slave_robot.py --master-ip <IP_DEL_MAESTRO>

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

# ─────────── parámetros del robot ────────────────────────────────────────────
L1, L2, L3 = 0.35, 0.30, 0.20
M1, M2, M3 = 1.5, 1.0, 0.5
G_GRAV      = 9.81
DT          = 0.01    # 100 Hz

# ─────────── impedancia ───────────────────────────────────────────────────────
KD_IMP = 170.0   # bajado 200→170: menos rigidez para no overshooting
BD_IMP = 90.0    # subido 70→90:   amortiguamiento alto mata la oscilación

# ─────────── Peg-in-Hole ─────────────────────────────────────────────────────
PEG_LENGTH  = 0.08
PEG_RADIUS  = 0.008
HOLE_CENTER = np.array([0.55, 0.20])   # más arriba → fácil de alcanzar desde init
HOLE_RADIUS = 0.009
F_CONTACT_K = 2000.0
F_THRESHOLD = 2.0

# ─────────── obstáculo ───────────────────────────────────────────────────────
OBS_CENTER = np.array([0.55, 0.36])   # justo arriba del HOLE (0.55, 0.20)
OBS_RADIUS = 0.04
OBS_K      = 250.0

Q_INIT = np.array([0.4, -0.3, 0.2])
Q_LIM  = np.array([np.pi/2, 2*np.pi/3, np.pi/2])


# ──────────────────────────────────────────────────────────────────────────────
# CINEMÁTICA
# ──────────────────────────────────────────────────────────────────────────────
def fk_3r(q):
    q1,q2,q3=q
    x1=L1*np.cos(q1);             y1=L1*np.sin(q1)
    x2=x1+L2*np.cos(q1+q2);      y2=y1+L2*np.sin(q1+q2)
    x3=x2+L3*np.cos(q1+q2+q3);   y3=y2+L3*np.sin(q1+q2+q3)
    return np.array([x3,y3])

def fk_3r_full(q):
    q1,q2,q3=q
    p0=np.zeros(2)
    p1=np.array([L1*np.cos(q1),L1*np.sin(q1)])
    p2=p1+np.array([L2*np.cos(q1+q2),L2*np.sin(q1+q2)])
    p3=p2+np.array([L3*np.cos(q1+q2+q3),L3*np.sin(q1+q2+q3)])
    return np.array([p0,p1,p2,p3])

def jacobian_3r(q):
    q1,q2,q3=q
    s1=np.sin(q1); s12=np.sin(q1+q2); s123=np.sin(q1+q2+q3)
    c1=np.cos(q1); c12=np.cos(q1+q2); c123=np.cos(q1+q2+q3)
    return np.array([
        [-L1*s1-L2*s12-L3*s123,-L2*s12-L3*s123,-L3*s123],
        [ L1*c1+L2*c12+L3*c123, L2*c12+L3*c123, L3*c123]
    ])


# ──────────────────────────────────────────────────────────────────────────────
# DINÁMICA
# ──────────────────────────────────────────────────────────────────────────────
def inertia_matrix(q):
    q1,q2,q3=q
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


# ──────────────────────────────────────────────────────────────────────────────
# PASO UNIFICADO DE IMPEDANCIA + INTEGRACIÓN
# M, C, g calculados UNA SOLA VEZ por paso
# ──────────────────────────────────────────────────────────────────────────────
def impedance_step(q, dq, x_des, F_contact=None, kd=KD_IMP, bd=BD_IMP, dt=DT):
    """
    Calcula M, C, g una vez y los reutiliza en control e integración.

    Returns:
        q_new, dq_new, tau, e_x
    """
    # ── Matrices dinámicas (una vez) ────────────────────────────────────
    M_mat = inertia_matrix(q)
    C_mat = coriolis_matrix(q, dq)
    g_vec = gravity_vector(q)

    # ── Impedancia ──────────────────────────────────────────────────────
    x_cur  = fk_3r(q)
    J      = jacobian_3r(q)
    dx_cur = J @ dq
    e_x    = np.clip(x_des - x_cur, -0.06, 0.06)   # ±6cm máx — evita saturación
    de_x   = -dx_cur   # dx_des = 0

    F_imp   = kd*e_x + bd*de_x
    F_total = F_imp.copy()
    if F_contact is not None:
        F_total += F_contact

    tau = J.T@F_total + g_vec + C_mat@dq
    tau = np.clip(tau, -20.0, 20.0)

    # ── Integración (reutiliza M, C, g) ─────────────────────────────────
    ddq    = np.linalg.solve(M_mat, tau - C_mat@dq - g_vec)
    dq_new = np.clip(dq + ddq*dt, -3.0, 3.0)
    q_new  = np.clip(q + dq_new*dt, -Q_LIM, Q_LIM)

    return q_new, dq_new, tau, e_x


# ──────────────────────────────────────────────────────────────────────────────
# OBSTÁCULO
# ──────────────────────────────────────────────────────────────────────────────
def obstacle_repulsion(x_ef):
    delta = x_ef - OBS_CENTER
    dist  = np.linalg.norm(delta)
    if dist < 1e-6:
        return np.array([1.0, 0.0])*OBS_K*OBS_RADIUS, True
    if dist < OBS_RADIUS:
        F_rep = OBS_K*(OBS_RADIUS-dist)*(delta/dist)
        return F_rep, True
    return np.zeros(2), False

def clamp_xdes_outside_obstacle(x_des):
    """
    Proyecta x_des al borde del obstáculo si cae dentro.
    Esto hace que el esclavo SE DETENGA en el borde
    en lugar de luchar contra la fuerza repulsiva.
    """
    delta = x_des - OBS_CENTER
    dist  = np.linalg.norm(delta)
    if dist < OBS_RADIUS and dist > 1e-6:
        x_des = OBS_CENTER + (delta/dist)*OBS_RADIUS
    return x_des


# ──────────────────────────────────────────────────────────────────────────────
# CONTACTO PEG-IN-HOLE
# ──────────────────────────────────────────────────────────────────────────────
class PegHoleContact:
    APPROACH, CONTACT, INSERTION, COMPLETE = 0, 1, 2, 3
    STATE_NAMES = {0:"APROXIMACIÓN", 1:"CONTACTO",
                   2:"INSERCIÓN",    3:"COMPLETADO ✓"}

    def __init__(self):
        self.phase = self.APPROACH
        self.depth = 0.0

    def reset(self):
        self.phase = self.APPROACH
        self.depth = 0.0

    def compute_contact_force(self, x_ef):
        delta = x_ef - HOLE_CENTER
        dist  = np.linalg.norm(delta)
        F_contact  = np.zeros(2)
        in_contact = False
        if abs(x_ef[1]-HOLE_CENTER[1]) < PEG_LENGTH*1.5:
            if dist < HOLE_RADIUS:
                self.phase = max(self.phase, self.INSERTION)
                self.depth = HOLE_CENTER[1]-x_ef[1]
                if self.depth > PEG_LENGTH*0.85:
                    self.phase = self.COMPLETE
            elif dist < HOLE_RADIUS+0.02:
                pen = dist-HOLE_RADIUS
                F_contact = -F_CONTACT_K*pen*(delta/dist)
                in_contact = True
                if self.phase < self.CONTACT:
                    self.phase = self.CONTACT
        return F_contact, self.STATE_NAMES[self.phase], in_contact


# ──────────────────────────────────────────────────────────────────────────────
# RED
# ──────────────────────────────────────────────────────────────────────────────
class SlaveNetServer:
    def __init__(self, master_ip="127.0.0.1", port_rx=9001):
        self.master_ip       = master_ip
        self.sock            = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', port_rx))
        self.sock.settimeout(0.005)
        self.x_des           = fk_3r(Q_INIT)   # empieza en posición inicial, no random
        self.gripper         = True
        self.reset_requested = False   # el maestro pidió reset
        self._reset_count    = 0       # paquetes de reset pendientes de enviar
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def request_reset(self):
        """Llama esto al presionar R en el esclavo: manda reset al maestro."""
        self._reset_count = 10

    def _recv_loop(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
                p = json.loads(data.decode())
                self.x_des   = np.array(p["xd"])
                self.gripper = bool(p["gripper"])
                # El maestro puede pedir reset al esclavo
                if p.get("reset", 0):
                    self.reset_requested = True
            except (socket.timeout, json.JSONDecodeError, AttributeError):
                pass

    def send_force(self, Fe, contact):
        do_reset = self._reset_count > 0
        if do_reset:
            self._reset_count -= 1
        msg = json.dumps({
            "Fe":      Fe.tolist(),
            "contact": int(contact),
            "reset":   int(do_reset)   # ← flag de reset al maestro
        })
        try:
            self.sock.sendto(msg.encode(), (self.master_ip, 9002))
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# ROBOT ESCLAVO
# ──────────────────────────────────────────────────────────────────────────────
class SlaveRobot:
    def __init__(self, master_ip="127.0.0.1"):
        self.net           = SlaveNetServer(master_ip)
        self.contact_model = PegHoleContact()
        self._reset_state()

        N = 500
        self.hist_t    = np.zeros(N)
        self.hist_tau  = np.zeros((N, 3))
        self.hist_Fc   = np.zeros((N, 2))
        self.hist_Fobs = np.zeros((N, 2))   # ← fuerza del obstáculo separada
        self.hist_x    = np.zeros((N, 2))
        self.hist_ex   = np.zeros((N, 2))
        self.idx = 0
        self.t   = 0.0
        self.contact_state = "APROXIMACIÓN"
        self.hitting_obs   = False
        self.resetting     = False
        self._settle       = 0
        self._F_obs_filt   = np.zeros(2)   # filtro paso bajo para F_obs
        self._x_des_filt   = fk_3r(Q_INIT).copy()  # filtro para x_des del maestro

    def _reset_state(self):
        self.q   = Q_INIT.copy()
        self.dq  = np.zeros(3)

    def _do_reset(self):
        """Lógica de reset pura (sin propagar por red)."""
        self._reset_state()
        self.net.x_des = fk_3r(Q_INIT)   # error de impedancia = 0
        self.contact_model.reset()
        N = 500
        self.hist_t[:]    = 0.0
        self.hist_tau[:]  = 0.0
        self.hist_Fc[:]   = 0.0
        self.hist_Fobs[:] = 0.0
        self.hist_x[:]    = 0.0
        self.hist_ex[:]   = 0.0
        self.idx           = 0
        self.t             = 0.0
        self.contact_state = "APROXIMACIÓN"
        self.hitting_obs   = False
        self.resetting     = True
        self._settle       = 50
        self._F_obs_filt   = np.zeros(2)
        self._x_des_filt   = fk_3r(Q_INIT).copy()  # reset del filtro

    def reset(self):
        """Reset iniciado por el usuario (tecla R): resetea y avisa al maestro."""
        self._do_reset()
        self.net.request_reset()   # propaga al maestro

    def step(self):
        # Si el maestro pidió reset, ejecutarlo SIN propagarlo de vuelta
        if self.net.reset_requested:
            self.net.reset_requested = False
            self._do_reset()   # ← no llama request_reset(), corta el loop
            return

        # Durante estabilización post-reset: ignorar al maestro, quedarse quieto
        if self._settle > 0:
            self._settle -= 1
            x_des = fk_3r(self.q)
            self._x_des_filt = x_des.copy()  # sincronizar filtro
        else:
            # Filtro paso bajo sobre x_des: α=0.35
            # Antes α=0.12 → lag de ~120mm, impedancia siempre saturada → oscilación
            # Ahora α=0.35 → lag ~20ms, el esclavo sigue sin quedarse atrás
            x_des_raw = clamp_xdes_outside_obstacle(self.net.x_des.copy())
            self._x_des_filt = 0.35 * x_des_raw + 0.65 * self._x_des_filt
            x_des = self._x_des_filt

        x_ef = fk_3r(self.q)

        # Fuerza de contacto con el agujero
        F_contact, state_str, in_contact = \
            self.contact_model.compute_contact_force(x_ef)
        self.contact_state = state_str

        # Fuerza repulsiva del obstáculo — filtrada para no causar spikes
        F_obs_raw, self.hitting_obs = obstacle_repulsion(x_ef)
        F_obs_raw = np.clip(F_obs_raw, -5.0, 5.0)
        # Filtro paso bajo α=0.3: suaviza el impacto inicial
        alpha = 0.3
        self._F_obs_filt = alpha * F_obs_raw + (1 - alpha) * self._F_obs_filt
        F_obs = self._F_obs_filt

        F_total_ctrl = F_contact + F_obs

        # Control + integración (M/C/g calculados UNA VEZ)
        self.q, self.dq, tau, e_x = impedance_step(
            self.q, self.dq, x_des, F_total_ctrl
        )

        # Enviar fuerza combinada al maestro
        self.net.send_force(F_total_ctrl, in_contact or self.hitting_obs)

        # Enviar telemetría al recorder (puerto 9003)
        try:
            telem = json.dumps({
                "src":   "slave",
                "t":     self.t,
                "q":     self.q.tolist(),
                "tau":   tau.tolist(),
                "Fc":    F_contact.tolist(),
                "Fobs":  F_obs.tolist(),
                "x":     x_ef.tolist(),
                "ex":    e_x.tolist(),
                "state": self.contact_state,
                "obs":   int(self.hitting_obs)
            })
            self.net.sock.sendto(telem.encode(), ("127.0.0.1", 9003))
        except Exception:
            pass

        i = self.idx % 500
        self.hist_t[i]    = self.t
        self.hist_tau[i]  = tau
        self.hist_Fc[i]   = F_contact
        self.hist_Fobs[i] = F_obs        # ← guardar fuerza del obstáculo
        self.hist_x[i]    = x_ef
        self.hist_ex[i]   = e_x
        self.idx += 1
        self.t   += DT


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────
def setup_slave_plots(robot):
    fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0f')
    fig.suptitle('TE3001B — Robot Esclavo 3R | Control de Impedancia + Peg-in-Hole   [R]=Reset',
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    bg = '#0d1117'
    C  = ['#FF6B6B','#69FF47','#00BFFF','#FFD700','#FF69B4','#00FFD0']

    ax_robot = fig.add_subplot(2,2,1, facecolor=bg)
    ax_force = fig.add_subplot(2,2,2, facecolor=bg)
    ax_tau   = fig.add_subplot(2,2,3, facecolor=bg)
    ax_err   = fig.add_subplot(2,2,4, facecolor=bg)

    for ax in [ax_robot, ax_force, ax_tau, ax_err]:
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        ax.grid(True, color='#1e2530', linestyle='--', alpha=0.5)
        ax.title.set_color('white')
        ax.xaxis.label.set_color('#aaa'); ax.yaxis.label.set_color('#aaa')

    # Panel 1
    ax_robot.set_xlim(-0.9, 0.9); ax_robot.set_ylim(-0.5, 1.0)
    ax_robot.set_aspect('equal')
    ax_robot.set_title('Esclavo — Peg-in-Hole', fontsize=11)
    ax_robot.set_xlabel('x [m]'); ax_robot.set_ylabel('y [m]')

    ax_robot.add_patch(plt.Rectangle(
        (HOLE_CENTER[0]-0.04, HOLE_CENTER[1]-0.005), 0.08, 0.04,
        color='#2a2a4a', zorder=1))
    ax_robot.plot(*HOLE_CENTER, 'x', color='#FFD700', markersize=10,
                  markeredgewidth=2, zorder=3)
    ax_robot.text(HOLE_CENTER[0]+0.02, HOLE_CENTER[1]+0.015,
                  'HOLE', color='#FFD700', fontsize=8)
    ax_robot.add_patch(plt.Circle(OBS_CENTER, OBS_RADIUS,
                                   color='#FF3333', alpha=0.85, zorder=2))
    ax_robot.text(OBS_CENTER[0]-0.03, OBS_CENTER[1]-0.005,
                  'OBS', color='white', fontsize=8, fontweight='bold', zorder=4)

    link_line, = ax_robot.plot([], [], 'o-', color=C[2], linewidth=3,
                                markersize=8, markerfacecolor=C[0])
    peg_line,  = ax_robot.plot([], [], '-',  color=C[3], linewidth=5, zorder=4)
    state_text  = ax_robot.text(0.02, 0.96, '', transform=ax_robot.transAxes,
                                 color='#FFD700', fontsize=10, fontweight='bold',
                                 verticalalignment='top')
    obs_text    = ax_robot.text(0.02, 0.88, '', transform=ax_robot.transAxes,
                                 color='#FF4444', fontsize=9, fontweight='bold',
                                 verticalalignment='top')
    reset_text  = ax_robot.text(0.5, 0.5, '', transform=ax_robot.transAxes,
                                 color='#00FF88', fontsize=16, fontweight='bold',
                                 ha='center', va='center',
                                 bbox=dict(boxstyle='round', facecolor='#0a0a0f',
                                           alpha=0.85, edgecolor='#00FF88'))

    # Panel 2: fuerzas — contacto (agujero) + obstáculo separados
    ax_force.set_title('Fuerzas: Contacto (sólido) vs Obstáculo (punteado) [N]', fontsize=10)
    ax_force.set_xlabel('Tiempo [s]'); ax_force.set_ylabel('F [N]')
    # Contacto con agujero
    line_Fx,  = ax_force.plot([], [], color='#FF69B4', linewidth=2.0, label='Fx contacto')
    line_Fy,  = ax_force.plot([], [], color='#00FFD0', linewidth=2.0, label='Fy contacto')
    # Obstáculo — mismos colores pero punteado
    line_Fox, = ax_force.plot([], [], color='#FF4444', linewidth=2.0,
                               linestyle='--', label='Fx obstáculo')
    line_Foy, = ax_force.plot([], [], color='#FF8800', linewidth=2.0,
                               linestyle='--', label='Fy obstáculo')
    ax_force.axhline(y=0, color='#444', linewidth=0.8)
    ax_force.axhline(y=F_THRESHOLD,  color='#FF4444', linewidth=1.0,
                     linestyle=':', alpha=0.6, label=f'Umbral {F_THRESHOLD} N')
    ax_force.legend(loc='upper right', fontsize=7,
                    facecolor='#1a1a2e', labelcolor='white')

    # Panel 3: torques
    ax_tau.set_title('Torques Articulares τ [Nm]', fontsize=11)
    ax_tau.set_xlabel('Tiempo [s]'); ax_tau.set_ylabel('τ [Nm]')
    lines_tau = [ax_tau.plot([], [], color=C[i], linewidth=1.5,
                              label=f'τ{i+1}')[0] for i in range(3)]
    ax_tau.axhline(y=0, color='#444', linewidth=0.8)
    ax_tau.legend(loc='upper right', fontsize=8,
                  facecolor='#1a1a2e', labelcolor='white')

    # Panel 4: error
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
            (link_line, peg_line, state_text, obs_text, reset_text),
            (line_Fx, line_Fy, line_Fox, line_Foy),
            lines_tau,
            (line_ex, line_ey))


def main(master_ip):
    robot = SlaveRobot(master_ip)
    (fig, axes, robot_artists, force_lines, lines_tau, err_lines) = \
        setup_slave_plots(robot)
    ax_robot, ax_force, ax_tau, ax_err = axes
    link_line, peg_line, state_text, obs_text, reset_text = robot_artists
    line_Fx, line_Fy, line_Fox, line_Foy = force_lines
    line_ex, line_ey = err_lines

    running      = [True]
    reset_frames = [0]

    def sim_loop():
        while running[0]:
            t0 = time.perf_counter()
            try:
                robot.step()
            except Exception as ex:
                print(f"[sim_loop] error: {ex} — reset automático")
                robot.reset()
            elapsed = time.perf_counter() - t0
            rem = DT - elapsed
            if rem > 0:
                time.sleep(rem)

    threading.Thread(target=sim_loop, daemon=True).start()

    def animate(frame):
        n  = min(robot.idx, 500)
        if n == 0:
            return [link_line, peg_line]
        i0  = robot.idx % 500
        idx = np.arange(i0, i0+n) % 500
        t   = robot.hist_t[idx]
        tau = robot.hist_tau[idx]
        Fc  = robot.hist_Fc[idx]
        Fo  = robot.hist_Fobs[idx]   # ← fuerza obstáculo
        ex  = robot.hist_ex[idx]
        t_win = 5.0
        mask  = (t > robot.t - t_win)

        pts = fk_3r_full(robot.q)
        link_line.set_data(pts[:,0], pts[:,1])
        ef = pts[-1]
        peg_dir = pts[-1]-pts[-2]
        if np.linalg.norm(peg_dir) > 0:
            peg_dir /= np.linalg.norm(peg_dir)
        peg_end = ef + peg_dir*PEG_LENGTH
        peg_line.set_data([ef[0], peg_end[0]], [ef[1], peg_end[1]])

        state_text.set_text(f"Estado: {robot.contact_state}")
        obs_text.set_text("⚠ OBSTÁCULO — bloqueado" if robot.hitting_obs else "")

        if robot.resetting:
            reset_frames[0] = 20
            robot.resetting = False
        if reset_frames[0] > 0:
            reset_text.set_text('RESET ✓')
            reset_frames[0] -= 1
        else:
            reset_text.set_text('')

        ax_force.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        line_Fx.set_data(t[mask], Fc[mask,0])
        line_Fy.set_data(t[mask], Fc[mask,1])
        line_Fox.set_data(t[mask], Fo[mask,0])   # ← obstáculo
        line_Foy.set_data(t[mask], Fo[mask,1])   # ← obstáculo
        ax_force.relim(); ax_force.autoscale_view(scalex=False)

        ax_tau.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        for i, ln in enumerate(lines_tau):
            ln.set_data(t[mask], tau[mask,i])
        ax_tau.relim(); ax_tau.autoscale_view(scalex=False)

        ax_err.set_xlim(max(0, robot.t-t_win), max(t_win, robot.t))
        line_ex.set_data(t[mask], np.abs(ex[mask,0])*1000)
        line_ey.set_data(t[mask], np.abs(ex[mask,1])*1000)
        ax_err.relim(); ax_err.autoscale_view(scalex=False)

        return ([link_line, peg_line, state_text, obs_text, reset_text] +
                lines_tau + [line_Fx, line_Fy, line_Fox, line_Foy, line_ex, line_ey])

    def on_key_press(event):
        if   event.key == 'r':      robot.reset()
        elif event.key == 'escape':
            running[0] = False
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    ani = animation.FuncAnimation(fig, animate, interval=50,
                                  blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TE3001B — Robot Esclavo")
    parser.add_argument("--master-ip", default="127.0.0.1")
    args = parser.parse_args()
    main(args.master_ip)