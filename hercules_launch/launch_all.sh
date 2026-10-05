#!/bin/bash
# ============================================================
# Lanza la campaña E2/E3/E4 (290 celdas) en Hércules.
# Vía: NAS directo (VPN CICA con openfortivpn) — NO requiere el MSI.
# Uso:  bash launch_all.sh
# ============================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ND="ssh -o BatchMode=yes hercules-cica"

echo "== 1) comprobando acceso =="
$ND hostname || { echo "ERROR: sin acceso. ¿VPN caída? → sudo hercules-vpn status"; exit 1; }

echo "== 2) subiendo scripts + lista de celdas =="
tar czf - -C "$HERE" bootstrap_compute.sh slurm_e234c.sh cmds_e234.txt \
  | $ND 'tar xzf - -C ~/crossdomain_qcl'

echo "== 3) lanzando bootstrap (job de cómputo: instala entorno y envía el array) =="
$ND 'cd ~/crossdomain_qcl && sbatch bootstrap_compute.sh'

echo "== 4) estado =="
$ND 'squeue -u dmartin | head -6'

cat <<'EOF'
------------------------------------------------------------
El job 'bootstrap' tarda ~10-15 min (miniconda + entorno + pip + datasets)
y al terminar envía el array de 290 celdas automáticamente.

Monitorizar:
  ssh hercules-cica 'squeue -u dmartin | head; tail -5 ~/crossdomain_qcl/bootstrap4_*.out'
------------------------------------------------------------
EOF
