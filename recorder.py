"""
╔══════════════════════════════════════════════════════════════╗
║  RECORDER — TE3001B  Peg-in-Hole Teleoperado                 ║
║  Graba telemetría de maestro y esclavo en tiempo real        ║
║  y genera gráficas de publicación al finalizar               ║
╚══════════════════════════════════════════════════════════════╝

Ejecutar en cualquier PC (misma red o loopback):
    python3 recorder.py

Flujo:
    1. Abre ventana con botones START / STOP
    2. Presiona START cuando quieras empezar a grabar
    3. Haz la demo (mueve el maestro, inserta el peg, esquiva obstáculo)
    4. Presiona STOP
    5. Se generan y guardan automáticamente las gráficas en ./graficas/
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import socket, threading, json, os, time
from datetime import datetime

# ─────────── configuración ────────────────────────────────────
PORT        = 9003          # puerto donde llegan las telemetrías
SAVE_DIR    = "graficas"    # carpeta de salida
os.makedirs(SAVE_DIR, exist_ok=True)

# Colores tema oscuro
BG    = '#0d1117'
FIG   = '#0a0a1a'
C_M   = ['#00BFFF', '#FF6B6B', '#69FF47']   # maestro τ1,τ2,τ3
C_S   = ['#FF6B6B', '#69FF47', '#00BFFF']   # esclavo τ1,τ2,τ3
C_FE  = ['#FF69B4', '#00FFD0']               # fuerzas reflejadas
C_FC  = ['#FF69B4', '#00FFD0']               # contacto
C_FO  = ['#FF4444', '#FF8800']               # obstáculo
C_Q   = ['#00BFFF', '#FF6B6B', '#69FF47']   # ángulos
C_ERR = ['#FF6B6B', '#69FF47']              # error


# ──────────────────────────────────────────────────────────────
# ALMACENAMIENTO DE DATOS
# ──────────────────────────────────────────────────────────────
class Recorder:
    def __init__(self):
        self.recording = False
        self.master = []   # lista de dicts
        self.slave  = []

    def start(self):
        self.master = []
        self.slave  = []
        self.recording = True
        print(f"[REC] ● Grabando...")

    def stop(self):
        self.recording = False
        print(f"[REC] ■ Detenido. "
              f"Maestro: {len(self.master)} muestras  "
              f"Esclavo: {len(self.slave)} muestras")

    def push(self, pkt):
        if not self.recording:
            return
        if pkt["src"] == "master":
            self.master.append(pkt)
        else:
            self.slave.append(pkt)

    def to_arrays(self):
        """Convierte listas de dicts a arrays numpy."""
        def extract(data, key, shape):
            try:
                arr = np.array([d[key] for d in data])
                return arr
            except (KeyError, ValueError):
                return np.zeros((len(data),) + shape)

        m, s = self.master, self.slave
        if len(m) == 0 or len(s) == 0:
            return None

        M = {
            "t":   extract(m, "t",   ()),
            "q":   extract(m, "q",   (3,)),
            "tau": extract(m, "tau", (3,)),
            "Fe":  extract(m, "Fe",  (2,)),
            "x":   extract(m, "x",   (2,)),
            "w":   extract(m, "w",   ()),
        }
        S = {
            "t":     extract(s, "t",     ()),
            "q":     extract(s, "q",     (3,)),
            "tau":   extract(s, "tau",   (3,)),
            "Fc":    extract(s, "Fc",    (2,)),
            "Fobs":  extract(s, "Fobs",  (2,)),
            "x":     extract(s, "x",     (2,)),
            "ex":    extract(s, "ex",    (2,)),
            "state": [d.get("state","") for d in s],
            "obs":   extract(s, "obs",   ()),
        }
        # Normalizar tiempo desde 0
        t0 = min(M["t"][0], S["t"][0])
        M["t"] -= t0
        S["t"] -= t0
        return M, S


# ──────────────────────────────────────────────────────────────
# RECEPTOR UDP
# ──────────────────────────────────────────────────────────────
def udp_listener(rec: Recorder):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    sock.settimeout(1.0)
    print(f"[REC] Escuchando en UDP:{PORT}")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            pkt = json.loads(data.decode())
            rec.push(pkt)
        except socket.timeout:
            pass
        except json.JSONDecodeError:
            pass


# ──────────────────────────────────────────────────────────────
# GENERACIÓN DE GRÁFICAS
# ──────────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="Tiempo [s]", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#aaa', labelsize=9)
    ax.xaxis.label.set_color('#aaa')
    ax.yaxis.label.set_color('#aaa')
    ax.title.set_color('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.grid(True, color='#1e2530', linestyle='--', alpha=0.6)
    if title:  ax.set_title(title, fontsize=11, pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


def add_contact_shading(ax, t, states, obs):
    """Sombrea fondos según estado de la tarea."""
    state_colors = {
        "CONTACTO":    ('#FFD700', 0.10),
        "INSERCIÓN":   ('#00FF88', 0.12),
        "COMPLETADO ✓":('#00BFFF', 0.15),
    }
    prev = None
    t0_s = t[0]
    for i, st in enumerate(states):
        if st != prev:
            if prev in state_colors and i > 0:
                c, a = state_colors[prev]
                ax.axvspan(t0_s, t[i], color=c, alpha=a)
            t0_s = t[i]
            prev = st
    if prev in state_colors:
        c, a = state_colors[prev]
        ax.axvspan(t0_s, t[-1], color=c, alpha=a)
    # Obstáculo
    in_obs = obs > 0.5
    if in_obs.any():
        starts = np.where(np.diff(in_obs.astype(int)) == 1)[0]
        ends   = np.where(np.diff(in_obs.astype(int)) == -1)[0]
        if in_obs[0]:  starts = np.concatenate([[0], starts])
        if in_obs[-1]: ends   = np.concatenate([ends, [len(t)-1]])
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], color='#FF3333', alpha=0.15)


def generate_plots(rec: Recorder):
    result = rec.to_arrays()
    if result is None:
        print("[REC] No hay datos suficientes para graficar.")
        return

    M, S = result
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── FIGURA 1: MAESTRO ──────────────────────────────────────────────────
    fig1 = plt.figure(figsize=(16, 10), facecolor=FIG)
    fig1.suptitle('TE3001B — Robot Maestro | Análisis de Telemetría',
                  color='white', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.42, wspace=0.35)

    # 1a: Trayectoria cartesiana
    ax = fig1.add_subplot(gs[0, 0])
    style_ax(ax, 'Trayectoria EF Maestro', 'x [m]', 'y [m]')
    sc = ax.scatter(M["x"][:,0], M["x"][:,1], c=M["t"],
                    cmap='plasma', s=4, alpha=0.8)
    ax.plot(M["x"][0,0], M["x"][0,1], 'o', color='#00FF88',
            markersize=10, label='Inicio', zorder=5)
    ax.plot(M["x"][-1,0], M["x"][-1,1], 's', color='#FF4444',
            markersize=10, label='Fin', zorder=5)
    plt.colorbar(sc, ax=ax, label='t [s]').ax.yaxis.label.set_color('#aaa')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax.set_aspect('equal')

    # 1b: Torques maestro
    ax = fig1.add_subplot(gs[0, 1])
    style_ax(ax, 'Torques Articulares Maestro', 'Tiempo [s]', 'τ [Nm]')
    for i in range(3):
        ax.plot(M["t"], M["tau"][:,i], color=C_M[i],
                linewidth=1.2, label=f'τ{i+1}')
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 1c: Fuerzas reflejadas (haptic)
    ax = fig1.add_subplot(gs[0, 2])
    style_ax(ax, 'Fuerzas Reflejadas — Haptic Feedback', 'Tiempo [s]', 'F [N]')
    ax.plot(M["t"], M["Fe"][:,0], color=C_FE[0], linewidth=1.5, label='Fx')
    ax.plot(M["t"], M["Fe"][:,1], color=C_FE[1], linewidth=1.5, label='Fy')
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 1d: Ángulos articulares maestro
    ax = fig1.add_subplot(gs[1, 0])
    style_ax(ax, 'Ángulos Articulares Maestro', 'Tiempo [s]', 'q [rad]')
    lims  = [np.pi/2, 2*np.pi/3, np.pi/2]
    names = ['q1','q2','q3']
    for i in range(3):
        ax.plot(M["t"], M["q"][:,i], color=C_Q[i],
                linewidth=1.2, label=names[i])
        ax.axhline( lims[i], color=C_Q[i], linewidth=0.6,
                    linestyle=':', alpha=0.5)
        ax.axhline(-lims[i], color=C_Q[i], linewidth=0.6,
                    linestyle=':', alpha=0.5)
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 1e: Manipulabilidad
    ax = fig1.add_subplot(gs[1, 1])
    style_ax(ax, 'Manipulabilidad w(q)', 'Tiempo [s]', 'w')
    ax.plot(M["t"], M["w"], color='#FF4444', linewidth=1.5)
    ax.axhline(0.05, color='#FF4444', linewidth=1.0,
               linestyle='--', alpha=0.7, label='Umbral singularidad')
    ax.fill_between(M["t"], 0, M["w"], alpha=0.2, color='#FF4444')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 1f: Norma de fuerza háptica
    ax = fig1.add_subplot(gs[1, 2])
    style_ax(ax, '|Fe| — Magnitud Feedback Háptico', 'Tiempo [s]', '|Fe| [N]')
    Fe_norm = np.linalg.norm(M["Fe"], axis=1)
    ax.plot(M["t"], Fe_norm, color='#FFD700', linewidth=1.5)
    ax.fill_between(M["t"], 0, Fe_norm, alpha=0.25, color='#FFD700')

    path1 = os.path.join(SAVE_DIR, f"maestro_{ts}.png")
    fig1.savefig(path1, dpi=150, bbox_inches='tight',
                 facecolor=FIG, edgecolor='none')
    print(f"[REC] Guardado: {path1}")

    # ── FIGURA 2: ESCLAVO ──────────────────────────────────────────────────
    fig2 = plt.figure(figsize=(16, 10), facecolor=FIG)
    fig2.suptitle('TE3001B — Robot Esclavo | Análisis de Telemetría',
                  color='white', fontsize=14, fontweight='bold')
    gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.35)

    # 2a: Trayectoria esclavo + HOLE + OBS
    ax = fig2.add_subplot(gs2[0, 0])
    style_ax(ax, 'Trayectoria EF Esclavo', 'x [m]', 'y [m]')
    sc = ax.scatter(S["x"][:,0], S["x"][:,1], c=S["t"],
                    cmap='viridis', s=4, alpha=0.8)
    ax.plot(S["x"][0,0], S["x"][0,1], 'o', color='#00FF88',
            markersize=9, label='Inicio', zorder=6)
    ax.plot(S["x"][-1,0], S["x"][-1,1], 's', color='#FF4444',
            markersize=9, label='Fin', zorder=6)
    ax.add_patch(plt.Circle((0.55, 0.20), 0.009,
                              color='#FFD700', alpha=0.8, zorder=5, label='HOLE'))
    ax.add_patch(plt.Circle((0.55, 0.36), 0.04,
                              color='#FF3333', alpha=0.5, zorder=4, label='OBS'))
    plt.colorbar(sc, ax=ax, label='t [s]').ax.yaxis.label.set_color('#aaa')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax.set_aspect('equal')

    # 2b: Torques esclavo + sombreado de estado
    ax = fig2.add_subplot(gs2[0, 1])
    style_ax(ax, 'Torques Articulares Esclavo', 'Tiempo [s]', 'τ [Nm]')
    add_contact_shading(ax, S["t"], S["state"], S["obs"])
    for i in range(3):
        ax.plot(S["t"], S["tau"][:,i], color=C_S[i],
                linewidth=1.2, label=f'τ{i+1}')
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 2c: Fuerzas de contacto y obstáculo
    ax = fig2.add_subplot(gs2[0, 2])
    style_ax(ax, 'Fuerzas: Contacto vs Obstáculo', 'Tiempo [s]', 'F [N]')
    add_contact_shading(ax, S["t"], S["state"], S["obs"])
    ax.plot(S["t"], S["Fc"][:,0],   color=C_FC[0], linewidth=1.5,
            label='Fx contacto')
    ax.plot(S["t"], S["Fc"][:,1],   color=C_FC[1], linewidth=1.5,
            label='Fy contacto')
    ax.plot(S["t"], S["Fobs"][:,0], color=C_FO[0], linewidth=1.5,
            linestyle='--', label='Fx obstáculo')
    ax.plot(S["t"], S["Fobs"][:,1], color=C_FO[1], linewidth=1.5,
            linestyle='--', label='Fy obstáculo')
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')

    # 2d: Error cartesiano
    ax = fig2.add_subplot(gs2[1, 0])
    style_ax(ax, 'Error Cartesiano |e|', 'Tiempo [s]', 'Error [mm]')
    add_contact_shading(ax, S["t"], S["state"], S["obs"])
    ex_mm = np.abs(S["ex"]) * 1000
    ax.plot(S["t"], ex_mm[:,0], color=C_ERR[0], linewidth=1.5, label='|eₓ|')
    ax.plot(S["t"], ex_mm[:,1], color=C_ERR[1], linewidth=1.5, label='|e_y|')
    ax.axhline(1.0, color='#888', linewidth=1.0,
               linestyle=':', label='1 mm (meta)')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 2e: Ángulos articulares esclavo
    ax = fig2.add_subplot(gs2[1, 1])
    style_ax(ax, 'Ángulos Articulares Esclavo', 'Tiempo [s]', 'q [rad]')
    for i in range(3):
        ax.plot(S["t"], S["q"][:,i], color=C_Q[i],
                linewidth=1.2, label=names[i])
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    # 2f: Norma de fuerza total + línea de fase
    ax = fig2.add_subplot(gs2[1, 2])
    style_ax(ax, '|F_contacto| + |F_obs| — Magnitudes', 'Tiempo [s]', '|F| [N]')
    Fc_norm   = np.linalg.norm(S["Fc"],   axis=1)
    Fobs_norm = np.linalg.norm(S["Fobs"], axis=1)
    ax.plot(S["t"], Fc_norm,   color='#FFD700', linewidth=1.5, label='|Fc| agujero')
    ax.plot(S["t"], Fobs_norm, color='#FF4444', linewidth=1.5,
            linestyle='--', label='|Fobs| obstáculo')
    ax.fill_between(S["t"], 0, Fc_norm,   alpha=0.2, color='#FFD700')
    ax.fill_between(S["t"], 0, Fobs_norm, alpha=0.2, color='#FF4444')
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')

    path2 = os.path.join(SAVE_DIR, f"esclavo_{ts}.png")
    fig2.savefig(path2, dpi=150, bbox_inches='tight',
                 facecolor=FIG, edgecolor='none')
    print(f"[REC] Guardado: {path2}")

    # ── FIGURA 3: COMPARATIVA BILATERAL ────────────────────────────────────
    fig3 = plt.figure(figsize=(16, 6), facecolor=FIG)
    fig3.suptitle('TE3001B — Comparativa Bilateral Maestro vs Esclavo',
                  color='white', fontsize=14, fontweight='bold')
    gs3 = gridspec.GridSpec(1, 3, figure=fig3, wspace=0.35)

    # 3a: Trayectorias superpuestas
    ax = fig3.add_subplot(gs3[0])
    style_ax(ax, 'Trayectorias EF', 'x [m]', 'y [m]')
    ax.plot(M["x"][:,0], M["x"][:,1], color='#00BFFF',
            linewidth=1.5, alpha=0.8, label='Maestro')
    ax.plot(S["x"][:,0], S["x"][:,1], color='#FF6B6B',
            linewidth=1.5, alpha=0.8, label='Esclavo')
    ax.add_patch(plt.Circle((0.55, 0.20), 0.009,
                              color='#FFD700', zorder=5))
    ax.add_patch(plt.Circle((0.55, 0.36), 0.04,
                              color='#FF3333', alpha=0.35, zorder=4))
    ax.set_aspect('equal')
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    # 3b: Torques τ1 de ambos
    ax = fig3.add_subplot(gs3[1])
    style_ax(ax, 'Torque τ₁ — Maestro vs Esclavo', 'Tiempo [s]', 'τ₁ [Nm]')
    ax.plot(M["t"], M["tau"][:,0], color='#00BFFF',
            linewidth=1.5, label='τ₁ maestro')
    ax.plot(S["t"], S["tau"][:,0], color='#FF6B6B',
            linewidth=1.5, alpha=0.8, label='τ₁ esclavo')
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    # 3c: Fe maestro vs Fc esclavo
    ax = fig3.add_subplot(gs3[2])
    style_ax(ax, 'Haptic: Fe maestro vs Fc esclavo', 'Tiempo [s]', 'F [N]')
    Fe_n = np.linalg.norm(M["Fe"], axis=1)
    Fc_n = np.linalg.norm(S["Fc"], axis=1)
    ax.plot(M["t"], Fe_n, color='#FF69B4', linewidth=1.5, label='|Fe| maestro')
    ax.plot(S["t"], Fc_n, color='#00FFD0', linewidth=1.5, label='|Fc| esclavo')
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')

    path3 = os.path.join(SAVE_DIR, f"bilateral_{ts}.png")
    fig3.savefig(path3, dpi=150, bbox_inches='tight',
                 facecolor=FIG, edgecolor='none')
    print(f"[REC] Guardado: {path3}")

    plt.show()
    print(f"\n[REC] ✓ 3 figuras guardadas en ./{SAVE_DIR}/")


# ──────────────────────────────────────────────────────────────
# INTERFAZ GRÁFICA — ventana de control
# ──────────────────────────────────────────────────────────────
def main():
    rec = Recorder()

    # Arrancar receptor UDP en hilo daemon
    t = threading.Thread(target=udp_listener, args=(rec,), daemon=True)
    t.start()

    # Ventana de control
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=FIG)
    fig.canvas.manager.set_window_title("TE3001B — Recorder")
    ax.set_facecolor(FIG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    # Título y estado
    ax.text(0.5, 0.85, 'TE3001B — Grabador de Telemetría',
            transform=ax.transAxes, color='white', fontsize=11,
            fontweight='bold', ha='center')
    status_text = ax.text(0.5, 0.65, '⬤  Esperando...', transform=ax.transAxes,
                           color='#888', fontsize=12, ha='center')
    count_text  = ax.text(0.5, 0.48, '', transform=ax.transAxes,
                           color='#aaa', fontsize=9, ha='center')
    ax.text(0.5, 0.08,
            'Puerto UDP 9003  |  Guardar en ./graficas/',
            transform=ax.transAxes, color='#555', fontsize=8, ha='center')

    # Botón START
    ax_start = fig.add_axes([0.12, 0.18, 0.32, 0.18])
    btn_start = Button(ax_start, '▶  START', color='#1a4a1a', hovercolor='#2a7a2a')
    btn_start.label.set_color('#00FF88')
    btn_start.label.set_fontsize(11)
    btn_start.label.set_fontweight('bold')

    # Botón STOP + PLOT
    ax_stop = fig.add_axes([0.56, 0.18, 0.32, 0.18])
    btn_stop = Button(ax_stop, '■  STOP', color='#4a1a1a', hovercolor='#7a2a2a')
    btn_stop.label.set_color('#FF4444')
    btn_stop.label.set_fontsize(11)
    btn_stop.label.set_fontweight('bold')

    def on_start(event):
        rec.start()
        status_text.set_text('⬤  GRABANDO...')
        status_text.set_color('#FF4444')
        count_text.set_text('')
        fig.canvas.draw_idle()

    def on_stop(event):
        rec.stop()
        n_m = len(rec.master)
        n_s = len(rec.slave)
        status_text.set_text('⬤  Detenido')
        status_text.set_color('#888')
        count_text.set_text(
            f'Maestro: {n_m} muestras  |  Esclavo: {n_s} muestras\n'
            f'Generando gráficas...'
        )
        fig.canvas.draw_idle()
        # Generar en hilo para no bloquear la UI
        threading.Thread(target=generate_plots, args=(rec,), daemon=False).start()

    btn_start.on_clicked(on_start)
    btn_stop.on_clicked(on_stop)

    # Actualizar contador mientras graba
    def tick(frame):
        if rec.recording:
            n_m = len(rec.master)
            n_s = len(rec.slave)
            count_text.set_text(f'M: {n_m}  |  S: {n_s} muestras')
            fig.canvas.draw_idle()

    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, tick, interval=500, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    main()