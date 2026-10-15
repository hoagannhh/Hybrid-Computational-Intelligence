"""PSO hyperparameter search with full train/validation fitness and caching."""
import json
import time
import numpy as np
import pandas as pd
from src.training.lstm import fit_lstm
from src.utils.experiment import save_json


def decode_particle(position, config):
    return {"hidden_dim": int(np.clip(round(position[0] / 16) * 16, 16, 128)),
            "dropout": float(np.clip(position[1], 0, 0.5)),
            "learning_rate": float(10 ** np.clip(position[2], -4, -2)),
            "num_layers": config["NUM_LAYERS"]}


def pso_search(data, config, device, output):
    # Chỉ truyền train/val vào tìm kiếm. Test không tham gia fitness hoặc early stopping.
    rng = np.random.default_rng(config["SEARCH_SEED"])
    count = config["PSO_PARTICLES"]
    low, high = np.array([16, 0, -4.0]), np.array([128, .5, -2.0])
    particles = rng.uniform(low, high, (count, 3))
    velocity = np.zeros_like(particles)
    personal, scores = particles.copy(), np.full(count, np.inf)
    global_best, global_score = None, float("inf")
    cache, records = {}, []
    start = time.perf_counter()
    for iteration in range(config["PSO_ITERATIONS"] + 1):
        if iteration:
            velocity = (.5 * velocity + 1.5 * rng.random(particles.shape) * (personal - particles)
                        + 1.5 * rng.random(particles.shape) * (global_best - particles))
            particles = np.clip(particles + velocity, low, high)
        for i, particle in enumerate(particles):
            params = decode_particle(particle, config)
            key = json.dumps(params, sort_keys=True)
            cached = key in cache
            if not cached:
                model, _, result = fit_lstm(data, params, config, device, config["SEARCH_SEED"],
                                            config["SEARCH_EPOCHS"], config["SEARCH_PATIENCE"], quiet=True)
                cache[key] = result["best_val_mse"]
                del model
            score = cache[key]
            if score < scores[i]:
                scores[i], personal[i] = score, particle.copy()
            if score < global_score:
                global_score, global_best = score, particle.copy()
            records.append({"iteration": iteration, "particle": i, **params, "val_mse": score,
                            "best_val_mse": global_score, "cached": cached})
        pd.DataFrame(records).to_csv(output / "pso_history.csv", index=False)
        print(f"PSO {iteration}/{config['PSO_ITERATIONS']}: best val MSE={global_score:.6f}")
    result = {"params": decode_particle(global_best, config), "val_mse": global_score,
              "search_seconds": time.perf_counter() - start, "unique_candidates": len(cache)}
    save_json(output / "pso_best.json", result)
    return result
