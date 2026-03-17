"""
segmentor.py
============
Executa a inferencia do Ilastik 1.4.x via ilastik.experimental.api.

O pipeline usa from_project_file() para carregar o modelo .ilp e
get_probabilities() para obter o mapa de probabilidades por pixel.

REQUISITOS DE AMBIENTE:
  - Ilastik instalado via conda (ilastik-forge)
  - Os dois os.environ abaixo DEVEM ser definidos antes de importar qualquer
    modulo do ilastik. Este modulo os define na importacao.

Referencia: https://github.com/ilastik/ilastik/blob/main/ilastik/experimental/api.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Variaveis de ambiente DEVEM ser definidas antes de qualquer import do ilastik
# ──────────────────────────────────────────────────────────────────────────────
_THREADS_KEY  = "LAZYFLOW_THREADS"
_RAM_KEY      = "LAZYFLOW_TOTAL_RAM_MB"

if _THREADS_KEY not in os.environ:
    os.environ[_THREADS_KEY] = "4"
if _RAM_KEY not in os.environ:
    os.environ[_RAM_KEY] = "4096"


class IlastikSegmentor:
    """
    Wrapper sobre ilastik.experimental.api para segmentacao de poros.

    Carrega o pipeline uma unica vez (from_project_file) e reutiliza
    para multiplas imagens — evita overhead de reinicializacao.

    Parameters
    ----------
    ilp_path     : caminho para o arquivo .ilp treinado
    n_threads    : numero de threads para lazyflow (default: 4)
    ram_mb       : RAM maxima para lazyflow em MB (default: 4096)
    pore_channel : indice do canal de saida que representa os poros.
                   Normalmente obtido de ILPMetadata.pore_class_index.

    Uso
    ---
    >>> seg = IlastikSegmentor("model.ilp", pore_channel=0)
    >>> prob_map = seg.predict(image)          # shape: (H, W, n_classes)
    >>> pore_prob = seg.pore_probability(image) # shape: (H, W)
    """

    def __init__(
        self,
        ilp_path: str | Path,
        n_threads: int = 4,
        ram_mb: int = 4096,
        pore_channel: int = 0,
    ) -> None:
        self.ilp_path     = Path(ilp_path).resolve()
        self.pore_channel = pore_channel

        # Atualiza variaveis de ambiente com parametros do usuario
        os.environ[_THREADS_KEY]  = str(n_threads)
        os.environ[_RAM_KEY]      = str(ram_mb)

        logger.info(
            f"Carregando pipeline Ilastik: {self.ilp_path} "
            f"(threads={n_threads}, RAM={ram_mb}MB)"
        )
        self._pipeline = self._load_pipeline()
        logger.info("Pipeline Ilastik carregado com sucesso.")

    # ------------------------------------------------------------------

    def _load_pipeline(self):
        """
        Carrega o pipeline usando ilastik.experimental.api.

        Esta importacao so pode ocorrer APOS os os.environ serem definidos.
        """
        try:
            from ilastik.experimental.api import from_project_file
        except ImportError as e:
            raise ImportError(
                "Nao foi possivel importar ilastik.experimental.api.\n"
                "Certifique-se de que o Ilastik esta instalado no ambiente conda ativo.\n"
                "Instrucoes: conda install -c ilastik-forge ilastik\n"
                f"Erro original: {e}"
            ) from e

        return from_project_file(str(self.ilp_path))

    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Executa a classificacao de pixels e retorna o mapa de probabilidades.

        Parameters
        ----------
        image : np.ndarray
            Imagem de entrada. Formatos aceitos:
              - (H, W)        : grayscale
              - (H, W, C)     : multicanal

            Tipo: float32 com valores em [0, 1].
            O preprocessamento (normalizacao) deve ser feito ANTES desta chamada.

        Returns
        -------
        np.ndarray
            Mapa de probabilidades com shape (H, W, n_classes), dtype float32.
            prob_map[:, :, i] = probabilidade de cada pixel pertencer a classe i.
        """
        image = self._prepare_input(image)

        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError(
                "xarray e necessario para a API do Ilastik.\n"
                "Instale com: pip install xarray\n"
                f"Erro: {e}"
            ) from e

        # ilastik.experimental.api espera xarray.DataArray com dims ["y","x","c"]
        # para imagens 2D.
        if image.ndim == 2:
            # grayscale sem canal: adiciona eixo c
            image = image[:, :, np.newaxis]

        # Confirma shape (H, W, C)
        assert image.ndim == 3, f"Esperado shape (H,W,C), obtido {image.shape}"

        input_xr = xr.DataArray(image, dims=["y", "x", "c"])

        logger.debug(
            f"Iniciando inferencia Ilastik — shape entrada: {image.shape}"
        )

        result = self._pipeline.get_probabilities(input_xr)

        # Converte para numpy se necessario
        prob_map = np.asarray(result, dtype=np.float32)

        # Garante shape (H, W, n_classes)
        if prob_map.ndim == 2:
            # Caso improvavel: saida escalar por pixel (binaria) → expande dim
            prob_map = prob_map[:, :, np.newaxis]

        logger.debug(f"Inferencia concluida — shape saida: {prob_map.shape}")
        return prob_map

    # ------------------------------------------------------------------

    def pore_probability(self, image: np.ndarray) -> np.ndarray:
        """
        Retorna apenas o canal de probabilidade do poro.

        Parameters
        ----------
        image : np.ndarray
            Mesmas regras de predict().

        Returns
        -------
        np.ndarray
            Shape (H, W), dtype float32, valores em [0, 1].
        """
        prob_map = self.predict(image)

        if self.pore_channel >= prob_map.shape[2]:
            raise ValueError(
                f"pore_channel={self.pore_channel} fora do intervalo: "
                f"o mapa tem {prob_map.shape[2]} canais. "
                "Corrija 'pore_class_index' no config.yaml."
            )

        return prob_map[:, :, self.pore_channel]

    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_input(image: np.ndarray) -> np.ndarray:
        """
        Valida e converte a imagem para float32.

        O Ilastik espera float32 normalizado — a normalizacao deve ter
        sido feita pelo preprocessador, mas garantimos o dtype aqui.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Esperado np.ndarray, obtido {type(image)}")
        if image.ndim not in (2, 3):
            raise ValueError(f"Imagem deve ser 2D ou 3D, obtido ndim={image.ndim}")

        image = np.asarray(image, dtype=np.float32)

        if image.max() > 1.0 + 1e-6:
            logger.warning(
                f"Imagem com valores acima de 1.0 (max={image.max():.3f}). "
                "Recomenda-se normalizar antes de chamar o segmentador."
            )
        return image
