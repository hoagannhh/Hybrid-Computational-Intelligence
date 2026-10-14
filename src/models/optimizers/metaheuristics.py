import numpy as np
from typing import Callable, Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ParticleSwarmOptimizer:
    """
    Thuật toán tối ưu hóa bầy đàn (Particle Swarm Optimization - PSO)
    Dùng để tìm siêu tham số tối ưu (Hyperparameter Tuning) hoặc trọng số mô hình lai.
    """
    def __init__(
        self,
        cost_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        num_particles: int = 20,
        max_iter: int = 30,
        w: float = 0.5,     # Trọng số quán tính (Inertia weight)
        c1: float = 1.5,    # Hệ số nhận thức cá nhân (Cognitive parameter)
        c2: float = 1.5     # Hệ số học hỏi xã hội (Social parameter)
    ):
        self.cost_func = cost_func
        self.dim = dim
        self.lower_bound, self.upper_bound = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self) -> Tuple[np.ndarray, float]:
        """Thực thi vòng lặp tối ưu hóa bầy đàn."""
        # Khởi tạo vị trí và vận tốc các hạt
        particles = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.num_particles, self.dim)
        )
        velocities = np.random.uniform(
            -(self.upper_bound - self.lower_bound) * 0.1,
            (self.upper_bound - self.lower_bound) * 0.1,
            (self.num_particles, self.dim)
        )

        p_best = particles.copy()
        p_best_scores = np.array([self.cost_func(p) for p in particles])

        best_idx = np.argmin(p_best_scores)
        g_best = p_best[best_idx].copy()
        g_best_score = p_best_scores[best_idx]

        logger.info(f"PSO bắt đầu tối ưu: {self.num_particles} hạt, {self.max_iter} thế hệ...")

        for it in range(self.max_iter):
            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Cập nhật vận tốc
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best[i] - particles[i])
                    + self.c2 * r2 * (g_best - particles[i])
                )

                # Cập nhật vị trí
                particles[i] = particles[i] + velocities[i]
                # Kẹp giới hạn biên
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                # Đánh giá hàm mục tiêu
                score = self.cost_func(particles[i])
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best[i] = particles[i].copy()

                    if score < g_best_score:
                        g_best_score = score
                        g_best = particles[i].copy()

            if (it + 1) % 5 == 0 or it == self.max_iter - 1:
                logger.info(f"Thế hệ {it + 1}/{self.max_iter} -> Điểm tối ưu hiện tại (Loss): {g_best_score:.6f}")

        return g_best, g_best_score
