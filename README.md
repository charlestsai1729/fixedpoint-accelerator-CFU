# Fixed-point accelerator CFU

## Clone repository
The repo should be cloned under `CFU-Playground/proj`.
```
cd CFU-Playground/proj
git clone https://github.com/charlestsai1729/fixedpoint-accelerator-CFU.git
```
## Run repository
```
make prog EXTRA_LITEX_ARGS="--cpu-variant=perf+cfu" USE_VIVADO=1 TTY=/dev/ttyUSB0
make load
```
## Modify repository
New CFU-Playground project can be added to the repo.
