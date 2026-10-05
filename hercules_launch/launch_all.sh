#!/bin/bash
# ============================================================
# Relanza la campaña E2/E3/E4 (290 celdas) en Hércules.
# Requisitos: MSI encendido (Tailscale) con la VPN del CICA ACTIVA.
# Uso:  bash launch_all.sh
# ============================================================
set -e
MSI=dani@100.95.88.12
ND="ssh -o BatchMode=yes"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1) comprobando vía a Hércules =="
$ND -o ConnectTimeout=8 "$MSI" "ssh -o BatchMode=yes hercules hostname" || {
  echo "ERROR: sin acceso. ¿MSI apagado o VPN del CICA caída?"; exit 1; }

echo "== 2) subiendo scripts v3 + lista de celdas =="
tar czf - -C "$HERE" bootstrap_hercules_v3.sh slurm_e234c.sh cmds_e234.txt \
  | $ND "$MSI" "ssh -o BatchMode=yes hercules 'tar xzf - -C ~/crossdomain_qcl'"

echo "== 3) lanzando bootstrap v3 (nohup: sobrevive a desconexiones del MSI) =="
$ND "$MSI" "ssh -o BatchMode=yes hercules 'cd ~/crossdomain_qcl && setsid nohup bash bootstrap_hercules_v3.sh >/dev/null 2>&1 < /dev/null & sleep 3; echo LANZADO; pgrep -f bootstrap_hercules_v3 | head -1'"

echo "== 4) estado inicial =="
$ND "$MSI" "ssh -o BatchMode=yes hercules 'tail -4 ~/crossdomain_qcl/bootstrap3.log 2>/dev/null; echo; squeue -u dmartin | head -5'"

cat <<'EOF'

------------------------------------------------------------
Listo. Secuencia esperada:
  ~1-2 min  descarga e instalación de Miniconda en $HOME
  ~2 min    creación del entorno (python 3.11)
  ~5-8 min  pip install (pennylane, torch, ...)
  ~1 min    descarga de datasets (MNIST/Fashion/KMNIST/CIFAR)
  -> luego   sbatch automático del array de 290 celdas

Monitorizar en cualquier momento:
  ssh dani@100.95.88.12 'ssh hercules "tail -20 ~/crossdomain_qcl/bootstrap3.log; squeue -u dmartin | head"'

Nota: los arrays corren en el clúster SIN necesitar el MSI.
El MSI solo hace falta para lanzar y para recoger resultados.
------------------------------------------------------------
EOF
