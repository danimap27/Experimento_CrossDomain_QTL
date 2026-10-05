# hercules_launch — Campaña E2/E3/E4 en Hércules

> Actualizado 5-oct-2026: **la VPN del CICA vive ahora en el NAS** (openfortivpn,
> servicio `hercules-vpn`) → el lanzamiento va **directo desde el NAS** con
> `ssh hercules-cica`. El MSI ya no es necesario.

## Relanzar el despliegue

```bash
bash launch_all.sh
```

Sube los scripts, lanza el job `bootstrap` (en un nodo de CÓMPUTO) que hace:
miniconda en `$HOME/miniconda3` → entorno `$HOME/envs/qcl` (py3.11) →
pip (pennylane/torch/...) → datasets → **array SLURM de 290 celdas**.

## Monitorizar

```bash
ssh hercules-cica 'squeue -u dmartin | head; tail -5 ~/crossdomain_qcl/bootstrap4_*.out'
ssh hercules-cica 'ls ~/crossdomain_qcl/results/ | grep -c e234'   # celdas completadas
```

## Hallazgos operativos del clúster (importantes)

1. **`/lustre` está montado `noexec` en los nodos de LOGIN.** Cualquier binario
   en `$HOME` (o módulos como Miniconda3) falla ahí con `Permission denied`.
   Toda instalación/ejecución debe ir por `sbatch`/`salloc` (los nodos de
   cómputo tienen exec **y** red). El bootstrap por eso es un job de cómputo.
2. El Miniconda del módulo del sistema solo es usable en cómputo; aun así,
   instalamos el nuestro en `$HOME/miniconda3` para tener control total.
3. La VPN del CICA (Fortinet, `bardo.cica.es:443`) corre en el NAS:
   `sudo hercules-vpn {start|stop|restart|status|log}` (sin password).
4. `~/.ssh/config` del NAS: alias `hercules-cica` (login.spc.cica.es, dmartin,
   clave `hercules_nas` autorizada en el clúster).

## Contenido

- `launch_all.sh` — relanzador completo (acceso → subida → bootstrap → estado).
- `bootstrap_compute.sh` — job SLURM que instala todo y envía el array.
- `slurm_e234c.sh` — plantilla del array (290 celdas, %48 concurrentes).
- `cmds_e234.txt` — lista de celdas (chain4 + smnist5/sfmnist5 + pair2 escalas
  + qubits×layers; ideal + heron_r2; sin scifar5 → pendiente decisión CIFAR).
