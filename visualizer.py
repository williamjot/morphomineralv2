"""
output/visualizer.py
====================
Geracao de visualizacoes da segmentacao e distribuicoes morfometricas.

Funcoes principais:
  - save_overlay        : imagem original + mascara de poros colorida
  - save_label_overlay  : cada poro com cor unica
  - save_probability_map: mapa de probabilidade em escala de cor
  - save_histograms     : histogramas das metricas PARTISAN
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend sem display — seguro para uso headless
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Gera e salva visualizacoes dos resultados de segmentacao.

    Parameters
    ----------
    output_dir : diretorio de saida
    dpi        : resolucao das figuras salvas
    cmap_overlay: colormap para o overlay de probabilidade
    """

    def __init__(
        self,
        output_dir: str | Path,
        dpi: int = 150,
        cmap_overlay: str = "jet",
    ) -> None:
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi         = dpi
        self.cmap_overlay = cmap_overlay

    # ------------------------------------------------------------------
    # Overlay basico
    # ------------------------------------------------------------------

    def save_overlay(
        self,
        image: np.ndarray,
        binary_mask: np.ndarray,
        stem: str = "overlay",
        alpha: float = 0.45,
        color: tuple = (1.0, 0.0, 0.0),
    ) -> Path:
        """
        Salva a imagem original com os poros coloridos em overlay.

        Parameters
        ----------
        image       : imagem original, qualquer dtype
        binary_mask : mascara bool (H, W)
        stem        : nome base do arquivo de saida
        alpha       : transparencia do overlay
        color       : cor RGB dos poros (default: vermelho)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=self.dpi)
        display = self._to_display(image)

        axes[0].imshow(display, cmap="gray" if display.ndim == 2 else None)
        axes[0].set_title("Imagem original", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(display, cmap="gray" if display.ndim == 2 else None)
        overlay = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
        overlay[binary_mask, 0] = color[0]
        overlay[binary_mask, 1] = color[1]
        overlay[binary_mask, 2] = color[2]
        overlay[binary_mask, 3] = alpha
        axes[1].imshow(overlay)

        porosity = 100.0 * binary_mask.sum() / binary_mask.size
        axes[1].set_title(
            f"Poros segmentados — porosidade: {porosity:.2f}%", fontsize=11
        )
        axes[1].axis("off")

        plt.tight_layout()
        path = self.output_dir / f"{stem}.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Overlay salvo: {path}")
        return path

    # ------------------------------------------------------------------
    # Overlay com labels individuais
    # ------------------------------------------------------------------

    def save_label_overlay(
        self,
        image: np.ndarray,
        label_map: np.ndarray,
        stem: str = "labels",
        alpha: float = 0.55,
    ) -> Path:
        """
        Salva overlay com cada poro em uma cor distinta.

        Parameters
        ----------
        image     : imagem original
        label_map : inteiros (H, W); 0 = fundo
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        display = self._to_display(image)
        ax.imshow(display, cmap="gray" if display.ndim == 2 else None)

        n_labels = int(label_map.max())
        if n_labels > 0:
            cmap     = plt.cm.get_cmap("tab20", n_labels)
            colored  = np.zeros((*label_map.shape, 4), dtype=np.float32)
            for lbl in range(1, n_labels + 1):
                mask          = label_map == lbl
                r, g, b, _    = cmap(lbl - 1)
                colored[mask] = [r, g, b, alpha]
            ax.imshow(colored)

        ax.set_title(f"Poros individuais (N={n_labels})", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        path = self.output_dir / f"{stem}.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Label overlay salvo: {path}")
        return path

    # ------------------------------------------------------------------
    # Mapa de probabilidade
    # ------------------------------------------------------------------

    def save_probability_map(
        self,
        prob_map: np.ndarray,
        stem: str = "probability",
    ) -> Path:
        """Salva o mapa de probabilidade de poros com escala de cor."""
        fig, ax = plt.subplots(figsize=(9, 7), dpi=self.dpi)
        im = ax.imshow(prob_map, cmap=self.cmap_overlay, vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label="Probabilidade de poro")
        ax.set_title("Mapa de probabilidade — classe poro", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        path = self.output_dir / f"{stem}.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Mapa de probabilidade salvo: {path}")
        return path

    # ------------------------------------------------------------------
    # Histogramas PARTISAN
    # ------------------------------------------------------------------

    def save_histograms(
        self,
        df: pd.DataFrame,
        metrics: Optional[list] = None,
        stem: str = "histograms",
        bins: int = 30,
    ) -> Path:
        """
        Salva uma figura com histogramas das metricas morfometricas.

        Parameters
        ----------
        df      : DataFrame retornado por run_partisan()
        metrics : lista de colunas para histogramar.
                  Se None, usa um subconjunto representativo padrao.
        bins    : numero de bins
        """
        if metrics is None:
            # Subconjunto mais informativos para poros
            candidates = [
                "A", "p", "CI_Circ", "CI_AR", "CI_Sol",
                "dH", "feret_major", "LL_Elo", "SC_Rec", "FF",
            ]
            metrics = [m for m in candidates if m in df.columns]

        n = len(metrics)
        if n == 0:
            logger.warning("Nenhuma metrica disponivel para histograma.")
            return None

        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=self.dpi)
        axes = np.array(axes).flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = df[metric].dropna()
            ax.hist(values, bins=bins, color="#3B8BD4", edgecolor="white", linewidth=0.5)
            ax.set_title(metric, fontsize=10)
            ax.set_xlabel("Valor", fontsize=9)
            ax.set_ylabel("Contagem", fontsize=9)
            ax.tick_params(labelsize=8)

            # Linha de mediana
            med = values.median()
            ax.axvline(med, color="#D85A30", linewidth=1.5,
                       label=f"median={med:.3f}")
            ax.legend(fontsize=8)

        # Esconde axes extras
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            f"Distribuicao das metricas PARTISAN — {len(df)} poros",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        path = self.output_dir / f"{stem}.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Histogramas salvos: {path}")
        return path

    # ------------------------------------------------------------------
    # Scatter plot: dois parametros
    # ------------------------------------------------------------------

    def save_scatter(
        self,
        df: pd.DataFrame,
        x_col: str = "CI_Circ",
        y_col: str = "CI_AR",
        color_col: Optional[str] = "A",
        stem: str = "scatter",
    ) -> Path:
        """Scatter plot de dois parametros morfometricos com colormap opcional."""
        if x_col not in df.columns or y_col not in df.columns:
            logger.warning(f"Colunas {x_col} ou {y_col} nao encontradas.")
            return None

        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

        c_vals = df[color_col] if color_col and color_col in df.columns else None
        sc = ax.scatter(
            df[x_col], df[y_col],
            c=c_vals, cmap="plasma",
            alpha=0.7, s=20, edgecolors="none",
        )
        if c_vals is not None:
            plt.colorbar(sc, ax=ax, label=color_col)

        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)
        ax.set_title(f"{x_col} vs {y_col} — {len(df)} poros", fontsize=11)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        plt.tight_layout()
        path = self.output_dir / f"{stem}.png"
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Scatter salvo: {path}")
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_display(image: np.ndarray) -> np.ndarray:
        """Converte imagem para float32 [0,1] apta para imshow."""
        img = np.asarray(image, dtype=np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        return img
