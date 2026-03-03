"""
net_test.py — Prueba de conectividad UDP entre maestro y esclavo.
Ejecutar en AMBAS computadoras para verificar la red antes del examen.

En PC-A (receptor):   python3 net_test.py --mode server
En PC-B (emisor):     python3 net_test.py --mode client --ip <IP-PC-A>
"""

import socket, time, argparse, statistics

PORT = 9999
N_PACKETS = 100   # número de paquetes de prueba


def run_server():
    """
    Servidor UDP: recibe paquetes y mide latencia de ida y vuelta (RTT).
    Responde inmediatamente con un eco para que el cliente mida RTT.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    print(f"[SERVIDOR] Esperando en puerto {PORT}...")
    count = 0
    while count < N_PACKETS:
        data, addr = sock.recvfrom(128)
        # Eco inmediato — el cliente mide el RTT
        sock.sendto(data, addr)
        count += 1
    print(f"[SERVIDOR] {count} paquetes procesados. ¡Conectividad OK!")
    sock.close()


def run_client(server_ip):
    """
    Cliente UDP: envía paquetes de prueba y mide RTT.
    Calcula estadísticas: mínimo, máximo, promedio, desviación estándar.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)   # timeout de 1 segundo por paquete
    rtts = []
    lost = 0

    print(f"[CLIENTE] Enviando {N_PACKETS} paquetes a {server_ip}:{PORT}...")
    for i in range(N_PACKETS):
        payload = f"PING_{i:04d}_{time.time():.6f}".encode()
        t0 = time.perf_counter()
        sock.sendto(payload, (server_ip, PORT))
        try:
            resp, _ = sock.recvfrom(128)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            rtts.append(rtt_ms)
        except socket.timeout:
            lost += 1
        time.sleep(0.01)   # 100 Hz

    sock.close()
    if rtts:
        print(f"\n{'='*50}")
        print(f"  Paquetes enviados  : {N_PACKETS}")
        print(f"  Paquetes perdidos  : {lost} ({100*lost/N_PACKETS:.1f}%)")
        print(f"  RTT mínimo         : {min(rtts):.2f} ms")
        print(f"  RTT máximo         : {max(rtts):.2f} ms")
        print(f"  RTT promedio       : {statistics.mean(rtts):.2f} ms")
        print(f"  Desv. estándar RTT : {statistics.stdev(rtts):.2f} ms")
        print(f"{'='*50}")
        # Evaluación para control de tiempo real
        if statistics.mean(rtts) < 10.0 and lost/N_PACKETS < 0.01:
            print("  ✓ RED APTA para control de impedancia (< 10 ms, < 1% pérdida)")
        else:
            print("  ✗ RED INADECUADA — revisar conexión WiFi o usar cable")
    else:
        print("ERROR: Sin respuesta del servidor. Verificar IP y firewall.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de red UDP")
    parser.add_argument("--mode", choices=["server", "client"],
                        required=True, help="Modo de ejecución")
    parser.add_argument("--ip", default="127.0.0.1",
                        help="IP del servidor (solo en modo cliente)")
    args = parser.parse_args()
    if args.mode == "server":
        run_server()
    else:
        run_client(args.ip)
