"""
output/reporter.py
==================
Geracao de relatorio PDF automatico com os resultados da segmentacao.

O relatorio inclui:
  - Capa com nome da imagem e data
  - Parametros de configuracao usados
  - Resumo de porosidade
  - Imagens inline (overlay, mapa de probabilidade, histogramas)
  - Tabela de estatisticas das metricas PARTISAN
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PDFReporter:
    """
    Gera um relatorio PDF da analise morfometrica de poros.

    Usa fpdf2 (FPDF2 >= 2.7).

    Parameters
    ----------
    output_dir : diretorio onde o PDF sera salvo
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def generate(
        self,
        image_name: str,
        porosity_pct: float,
        n_pores: int,
        df: pd.DataFrame,
        stats_df: Optional[pd.DataFrame] = None,
        image_paths: Optional[Dict[str, Path]] = None,
        config: Optional[dict] = None,
        stem: str = "report",
    ) -> Path:
        """
        Gera o relatorio PDF completo.

        Parameters
        ----------
        image_name   : nome da imagem analisada
        porosity_pct : porosidade total em %
        n_pores      : numero de poros analisados
        df           : DataFrame com resultados PARTISAN
        stats_df     : DataFrame de estatisticas descritivas (opcional)
        image_paths  : dict com caminhos para figuras
                       ex.: {"overlay": Path(...), "histograms": Path(...)}
        config       : dict com parametros usados (opcional)
        stem         : nome base do arquivo PDF

        Returns
        -------
        Path : caminho do PDF gerado
        """
        try:
            from fpdf import FPDF
        except ImportError:
            raise ImportError(
                "fpdf2 nao instalado. Execute: pip install fpdf2"
            )

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ── Capa ──────────────────────────────────────────────────────
        pdf.add_page()
        self._add_title_page(pdf, image_name, porosity_pct, n_pores)

        # ── Configuracao ──────────────────────────────────────────────
        if config:
            pdf.add_page()
            self._add_config_section(pdf, config)

        # ── Resumo de porosidade ──────────────────────────────────────
        pdf.add_page()
        self._add_summary_section(pdf, df, porosity_pct, n_pores)

        # ── Figuras ──────────────────────────────────────────────────
        if image_paths:
            for title, img_path in image_paths.items():
                if img_path and Path(img_path).exists():
                    pdf.add_page()
                    self._add_figure(pdf, Path(img_path), title)

        # ── Tabela de estatisticas ────────────────────────────────────
        if stats_df is not None and not stats_df.empty:
            pdf.add_page()
            self._add_stats_table(pdf, stats_df)

        path = self.output_dir / f"{stem}.pdf"
        pdf.output(str(path))
        logger.info(f"Relatorio PDF salvo: {path}")
        return path

    # ------------------------------------------------------------------
    # Secoes internas
    # ------------------------------------------------------------------

    def _add_title_page(
        self,
        pdf,
        image_name: str,
        porosity_pct: float,
        n_pores: int,
    ) -> None:
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, "Relatorio de Segmentacao de Poros", ln=True, align="C")
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "EDS / CBS / QEMSCAN — Analise Morfometrica PARTISAN", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Helvetica", size=12)
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        info_lines = [
            f"Imagem analisada : {image_name}",
            f"Data de geracao  : {now}",
            f"Porosidade total : {porosity_pct:.3f}%",
            f"Poros analisados : {n_pores}",
            "",
            "Gerado por: pore_segmentation (Ilastik 1.4 + PARTISAN v2.0)",
        ]
        for line in info_lines:
            pdf.cell(0, 8, line, ln=True)

    def _add_config_section(self, pdf, config: dict) -> None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Parametros de Configuracao", ln=True)
        pdf.set_font("Helvetica", size=9)
        pdf.set_fill_color(245, 245, 245)

        def _flatten(d: dict, prefix: str = "") -> List[str]:
            lines = []
            for k, v in d.items():
                key = f"{prefix}{k}"
                if isinstance(v, dict):
                    lines.extend(_flatten(v, prefix=key + "."))
                else:
                    lines.append(f"  {key}: {v}")
            return lines

        for line in _flatten(config):
            pdf.cell(0, 6, line, ln=True, fill=True)

    def _add_summary_section(
        self,
        pdf,
        df: pd.DataFrame,
        porosity_pct: float,
        n_pores: int,
    ) -> None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Resumo da Segmentacao", ln=True)
        pdf.set_font("Helvetica", size=11)

        lines = [
            f"Total de poros detectados : {n_pores}",
            f"Porosidade total          : {porosity_pct:.4f}%",
        ]

        if not df.empty and "area_px" in df.columns:
            lines += [
                f"Area media dos poros      : {df['area_px'].mean():.1f} px",
                f"Area mediana dos poros    : {df['area_px'].median():.1f} px",
                f"Maior poro               : {df['area_px'].max():.0f} px",
                f"Menor poro               : {df['area_px'].min():.0f} px",
            ]

        if not df.empty and "CI_Circ" in df.columns:
            lines += [
                "",
                "Metricas PARTISAN — medias:",
                f"  CI_Circ (circularidade)   : {df['CI_Circ'].mean():.4f}",
                f"  CI_AR   (razao aspecto)   : {df['CI_AR'].mean():.4f}",
                f"  CI_Sol  (solidez)         : {df['CI_Sol'].mean():.4f}",
                f"  FF      (form factor)     : {df['FF'].mean():.4f}",
            ]

        for line in lines:
            pdf.cell(0, 7, line, ln=True)

    def _add_figure(self, pdf, img_path: Path, title: str) -> None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, title.replace("_", " ").title(), ln=True)
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        try:
            pdf.image(str(img_path), w=usable_w)
        except Exception as e:
            logger.warning(f"Nao foi possivel incluir imagem '{img_path}': {e}")

    def _add_stats_table(self, pdf, stats_df: pd.DataFrame) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Estatisticas Descritivas — Metricas PARTISAN", ln=True)
        pdf.set_font("Helvetica", size=7)

        # Cabecalho
        col_w = [30, 16, 18, 16, 18, 16, 16]
        headers = ["Metrica", "Count", "Mean", "Std", "Median", "Min", "Max"]
        pdf.set_fill_color(200, 220, 240)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 6, h, border=1, fill=True, align="C")
        pdf.ln()

        # Linhas
        fill = False
        pdf.set_fill_color(235, 245, 255)
        for _, row in stats_df.iterrows():
            vals = [
                str(row.get("metric", "")),
                f"{row.get('count', 0):.0f}",
                f"{row.get('mean', 0):.4f}",
                f"{row.get('std', 0):.4f}",
                f"{row.get('median', 0):.4f}",
                f"{row.get('min', 0):.4f}",
                f"{row.get('max', 0):.4f}",
            ]
            for i, v in enumerate(vals):
                pdf.cell(col_w[i], 5, v, border=1, fill=fill, align="C")
            pdf.ln()
            fill = not fill
