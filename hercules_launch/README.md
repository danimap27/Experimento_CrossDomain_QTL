# hercules_launch — lanzamiento de la campaña E2/E3/E4 en Hércules

> Preparado el 5-oct-2026 tras la sesión de despliegue. El MSI de Dani se apagó
> a mitad del lanzamiento; **no quedó nada corriendo en Hércules** (el único
> array enviado llevaba un entorno roto y fue cancelado) y todo está listo para
> relanzar con un solo comando.

## Relanzar (cuando el MSI esté encendido con la VPN del CICA activa)

```bash
bash launch_all.sh
```

Eso hace, en orden: comprobar la vía → subir los scripts → lanzar el bootstrap
v3 con `nohup` (sigue aunque se apague el MSI) → mostrar estado. Después, solo,
sin más interacción:

1. Instala Miniconda en `$HOME/miniconda3` (el Miniconda del módulo del sistema
   no es usable: FS sin exec → `Permission denied`).
2. Crea el entorno `$HOME/envs/qcl` (python 3.11).
3. `pip install pennylane==0.44.1 torch torchvision numpy scikit-learn matplotlib`.
4. Descarga datasets a `./data` (MNIST, FashionMNIST, KMNIST, CIFAR-10).
5. Envía el **array SLURM de 290 celdas** (`slurm_e234c.sh` → `cmds_e234.txt`):
   chain4 (TIL/CIL), smnist5/sfmnist5 (class-IL), pair2 escala {500,2k,12k},
   qubits {4,6,8} × layers {2,3,4}, perfiles ideal + heron_r2.

## Monitorizar

```bash
ssh dani@100.95.88.12 'ssh hercules "tail -20 ~/crossdomain_qcl/bootstrap3.log; squeue -u dmartin | head"'
```

## Qué hay ya desplegado en Hércules (`~/crossdomain_qcl`)

- Repo completo (código E1 + `e234_runner.py` + `self_tests.py` + `core/`).
- `cmds_e234.txt` (290 celdas), `slurm_e234.sh`, `bootstrap_hercules.sh` (v1, roto),
  `bootstrap_hercules_v2.sh` (v2, roto), `slurm_e234b.sh`.

## Pendiente menor

- `scifar5` (split-CIFAR-10) excluido de la lista: se lanzará si el equipo
  confirma CIFAR en la batería principal (y tras `--extract-cifar`).
- E5 (mecanismo) es un lanzamiento aparte, mismo patrón.
