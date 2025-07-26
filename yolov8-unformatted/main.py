import torch
import warnings
import time
from pathlib import Path
import pandas as pd
from ultralytics.utils import LOGGER
from distill_model import YOLOv8Distillation

# ==================== GLOBAL SETTINGS ====================

MODEL_CONFIGS = [
    r"D:\YOLOV8-tomatod\parser\community\cfg\detect\effnet\yolov8n.yaml",
]

DATASET_CONFIGS = [
    {
        "data":    r"D:\YOLOV8-tomatod\tomatOD_yolo\data.yaml",
        "name":    "TomatOD",
        "teacher": r"D:\YOLOV8-tomatod\[FINAL] Baseline Results\YOLOv8l_Results\runs6\detect6\tomatOD_run\weights\best.pt"
    },
]

# scalar “constants” ----------------------------------------------------------
EPOCHS      = 10
BATCH       = 32
IMGSZ       = 320
LR0         = 0.01
LRF         = 0.0001
MOMENTUM    = 0.9
OPTIMIZER   = "SGD"
SAVE_PERIOD = 50
VAL         = True

# --- Start of New/Modified Code ---
# New static scaling factors to balance the distillation loss components
# Based on logs, feature_loss (~0.0001) needs to be scaled up to match logit_loss (~6.0)
FEATURE_LOSS_SCALE = 60000.0
LOGIT_LOSS_SCALE   = 1.0
# --- End of New/Modified Code ---

# master training-options dict ------------------------------------------------
TRAINING_PARAMS = {
    "epochs":      EPOCHS,
    "imgsz":       IMGSZ,
    "batch":       BATCH,
    "lr0":         LR0,
    "lrf":         LRF,
    "momentum":    MOMENTUM,
    "optimizer":   OPTIMIZER,
    "device":      0 if torch.cuda.is_available() else "cpu",
    "project":     "YOLOv8_KD",
    "save_period": SAVE_PERIOD,
    "val":         VAL,
    "plots":       True,
    "save":        True,
    "exist_ok":    True,
    "workers":     0,  # force single-threaded loaders on Windows
}

# Hyperparameter sweep settings
EPOCHS_SWEEP         = 10
DISTILL_WEIGHTS      = [0.1, 0.3, 0.5, 0.7, 0.9]
TEMPERATURES         = [2.0, 4.0, 6.0, 8.0]
FEATURE_LOSS_WEIGHTS = [0.2, 0.5, 0.8] # Sweep this to find the best balance


# ==================== HELPERS ====================

def get_model_name(yaml_path):
    return Path(yaml_path).stem

def get_epoch_weights(weights_dir, save_period, total_epochs):
    wd = Path(weights_dir)
    files = []
    for e in range(save_period, total_epochs + 1, save_period):
        p = wd / f"epoch{e}.pt"
        if p.exists(): files.append((e, str(p)))
    for tag in ["best", "last"]:
        p = wd / f"{tag}.pt"
        if p.exists(): files.append((tag, str(p)))
    return files

def test_model_weights(model_cfg, weight_path, epoch_id, dw, temp, flw, ds_cfg):
    try:
        ds_name = ds_cfg["name"]
        mdl = YOLOv8Distillation(weight_path)
        results = mdl.val(data=ds_cfg["data"])
        # safe extraction
        final_map   = getattr(results.box, "map",   0.0)
        final_map50 = getattr(results.box, "map50", 0.0)
        final_map75 = getattr(results.box, "map75", 0.0)

        out = {
            "dataset":             ds_name,
            "backbone":            get_model_name(model_cfg),
            "epoch":               epoch_id,
            "distill_weight":      dw,
            "temperature":         temp,
            "feature_loss_weight": flw,
            "feature_loss_scale":  FEATURE_LOSS_SCALE,
            "logit_loss_scale":    LOGIT_LOSS_SCALE,
            "teacher":             ds_cfg["teacher"],
            "mAP":                 final_map,
            "mAP50":               final_map50,
            "mAP75":               final_map75,
            "weights":             weight_path,
            "status":              "success",
        }
        del mdl
        torch.cuda.empty_cache()
        return out

    except Exception as e:
        return {
            "dataset":             ds_cfg["name"],
            "backbone":            get_model_name(model_cfg),
            "epoch":               epoch_id,
            "distill_weight":      dw,
            "temperature":         temp,
            "feature_loss_weight": flw,
            "feature_loss_scale":  FEATURE_LOSS_SCALE,
            "logit_loss_scale":    LOGIT_LOSS_SCALE,
            "teacher":             ds_cfg["teacher"],
            "mAP":                 0.0,
            "mAP50":               0.0,
            "mAP75":               0.0,
            "weights":             weight_path,
            "status":              f"failed: {e}",
        }

def save_results(row, ds_name, mdl_name, phase="sweep"):
    base = Path.cwd() / TRAINING_PARAMS["project"] / ds_name / mdl_name / phase
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / f"{phase}_results.csv"

    df_new = pd.DataFrame([row])

    if csv_path.exists():
        try:
            df_existing = pd.read_csv(csv_path)
            # Reorder columns to match existing if possible
            df_new = df_new[df_existing.columns.intersection(df_new.columns).tolist()]
        except Exception as e:
            LOGGER.warning(f"Could not align with existing CSV {csv_path}: {e}")
    
    try:
        df_new.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    except PermissionError as err:
        LOGGER.error(f"Permission denied saving results to {csv_path}: {err}")

    return csv_path


def check_teachers():
    missing = []
    for d in DATASET_CONFIGS:
        if not Path(d["teacher"]).exists():
            missing.append(d["teacher"])
    if missing:
        LOGGER.error(f"Missing teacher weights: {missing}")
        return False
    return True


# ==================== PHASE 1: HYPERPARAMETER SWEEP ====================

def find_best_params(model_cfg, ds_cfg):
    LOGGER.info(f"Sweep for {model_cfg} on {ds_cfg['name']}")
    best = None
    for dw in DISTILL_WEIGHTS:
        for temp in TEMPERATURES:
            for flw in FEATURE_LOSS_WEIGHTS:
                LOGGER.info(f" Testing dw={dw}, temp={temp}, flw={flw}")
                mdl = YOLOv8Distillation(model_cfg)
                args = {
                    **TRAINING_PARAMS,
                    "data":                ds_cfg["data"],
                    "name":                f"{ds_cfg['name']}/{get_model_name(model_cfg)}/sweep/dw{dw}_t{temp}_flw{flw}",
                    "epochs":              EPOCHS_SWEEP,
                    "distill_weight":      dw,
                    "temperature":         temp,
                    "teacher_weights":     ds_cfg["teacher"],
                    "feature_loss_weight": flw,
                    "feature_loss_scale":  FEATURE_LOSS_SCALE,
                    "logit_loss_scale":    LOGIT_LOSS_SCALE,
                }
                mdl.train(**args)
                wdir = Path(args["project"]) / args["name"] / "weights" / "best.pt"
                if wdir.exists():
                    res = test_model_weights(model_cfg, str(wdir), "best", dw, temp, flw, ds_cfg)
                    save_results(res, ds_cfg["name"], get_model_name(model_cfg), "sweep")
                    if res["status"] == "success" and (best is None or res["mAP"] > best["mAP"]):
                        best = res
                del mdl
                torch.cuda.empty_cache()
    return best


# ==================== PHASE 2: FULL TRAINING ====================

def full_train(model_cfg, ds_cfg, best):
    LOGGER.info(
        f"Full train {model_cfg} on {ds_cfg['name']} w/ "
        f"dw={best['distill_weight']}, temp={best['temperature']}, flw={best['feature_loss_weight']}"
    )
    mdl = YOLOv8Distillation(model_cfg)
    args = {
        **TRAINING_PARAMS,
        "data":                ds_cfg["data"],
        "name":                f"{ds_cfg['name']}/{get_model_name(model_cfg)}/final",
        "distill_weight":      best["distill_weight"],
        "temperature":         best["temperature"],
        "teacher_weights":     ds_cfg["teacher"],
        "feature_loss_weight": best["feature_loss_weight"],
        "feature_loss_scale":  best["feature_loss_scale"],
        "logit_loss_scale":    best["logit_loss_scale"],
    }
    start = time.time()
    mdl.train(**args)
    duration = (time.time() - start) / 60

    wdir   = Path(args["project"]) / args["name"] / "weights"
    epochs = get_epoch_weights(wdir, TRAINING_PARAMS["save_period"], TRAINING_PARAMS["epochs"])
    all_results = []
    for eid, w in epochs:
        r = test_model_weights(model_cfg, w, eid,
                               best["distill_weight"],
                               best["temperature"],
                               best["feature_loss_weight"],
                               ds_cfg)
        r["train_time_min"] = duration
        save_results(r, ds_cfg["name"], get_model_name(model_cfg), "final")
        all_results.append(r)
    del mdl
    torch.cuda.empty_cache()
    return all_results


# ==================== ORCHESTRATOR ====================

def run_pipeline():
    if not check_teachers():
        raise FileNotFoundError("Missing teacher files")
    summary = []
    total   = len(MODEL_CONFIGS) * len(DATASET_CONFIGS)
    n = 0

    LOGGER.info(f"▶️ Starting pipeline: {total} combos")
    for m in MODEL_CONFIGS:
        for d in DATASET_CONFIGS:
            n += 1
            LOGGER.info(f"{n}/{total} → Model={m}, Dataset={d['name']}")
            best = find_best_params(m, d)
            if best is None:
                LOGGER.error(" Sweep failed, skipping full train.")
                continue
            final = full_train(m, d, best)
            summary += final

    # Save summary CSV
    df  = pd.DataFrame(summary)
    out = Path.cwd() / TRAINING_PARAMS["project"] / "summary"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "all_results.csv", index=False)
    LOGGER.info("Pipeline complete. Results in %s", out)


if __name__ == "__main__":
    run_pipeline()