# RASI 1.0.0

Contrast model

    mu(a, t) + beta(p) + delta(p, a)

Ridge fit on a csv with participant, drug, time, contrast. concentration and baseline are used when the file has them. Reserved contrasts score each round. A contrast penalty and a retrieval check decide which rollouts enter the fit. The audit keeps a round only when pair error improves by more than its standard error. Inference serves that round.

C++ is the engine. Go starts the mesh. Python is the baselines and the modality adapters, standard library only.

## Devices

Train on 32 NVIDIA B200, one mask per device. Infer on 2 B200: device 0 holds drug-by-time means, device 1 holds the person terms, the coordinator adds them. CUDA 12.8+, sm_100. If those devices are not visible, pass --cpu-mesh 2. A GPU launch stops unless nvidia-smi -L shows 32 for train or 2 for infer.

## Build

g++ 12 (C++17) and Go 1.23.

    make

Binaries land in cpp/build/rasi and go/build/rasi-coord.

## Run

    ./scripts/train_32xb200.sh
    ./scripts/train_32xb200.sh --cpu-mesh 2
    ./cpp/build/rasi train --cohort data/e1_qt_cohort.csv
    ./cpp/build/rasi test --cohort data/e1_qt_cohort.csv --masks 2 --rounds 6
    ./cpp/build/rasi check
    ./scripts/infer_2xb200.sh --cpu-mesh 2
    ./cpp/build/rasi apply --cohort data/e1_qt_cohort.csv

configs/train_32xb200.json is 32 masks, 1 seed, 15 rounds, 5 rollouts, lambda 1, linear model.

    python3 python/baselines.py apply --cohort data/e1_qt_cohort.csv
    python3 python/modalities.py apply --cohort data/e1_qt_cohort.csv

Python 3.10+.
