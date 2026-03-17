"""
output/exporter.py
==================
Exporta os resultados da analise morfometrica para CSV, Excel e JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResultExporter:
    """
    Exporta DataFrames de resultados morfometricos.

    Parameters
    ----------
    output_dir : diretorio de saida (criado se nao existir)
    formats    : lista com "csv", "excel", "json"
    """

    def __init__(
        self,
        output_dir: str | Path,
        formats: List[str] = ("csv", "excel"),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = [f.lower() for f in formats]

    def export(
        self,
        df: pd.DataFrame,
        stem: str = "pore_analysis",
        stats_df: pd.DataFrame = None,
    ) -> dict:
        """
        Exporta o DataFrame de resultados.

        Parameters
        ----------
        df       : DataFrame principal (uma linha por poro)
        stem     : nome base dos arquivos (sem extensao)
        stats_df : DataFrame de estatisticas (opcional — exportado junto)

        Returns
        -------
        dict com paths dos arquivos gerados, keyed por formato
        """
        exported = {}

        if "csv" in self.formats:
            path = self._export_csv(df, stem)
            exported["csv"] = path
            if stats_df is not None and not stats_df.empty:
                stats_path = self._export_csv(stats_df, f"{stem}_stats")
                exported["csv_stats"] = stats_path

        if "excel" in self.formats:
            path = self._export_excel(df, stats_df, stem)
            exported["excel"] = path

        if "json" in self.formats:
            path = self._export_json(df, stem)
            exported["json"] = path

        return exported

    def _export_csv(self, df: pd.DataFrame, stem: str) -> Path:
        path = self.output_dir / f"{stem}.csv"
        df.to_csv(path, index=False, float_format="%.6f")
        logger.info(f"CSV salvo: {path}")
        return path

    def _export_excel(
        self,
        df: pd.DataFrame,
        stats_df: pd.DataFrame,
        stem: str,
    ) -> Path:
        path = self.output_dir / f"{stem}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Pores", index=False)
            if stats_df is not None and not stats_df.empty:
                stats_df.to_excel(writer, sheet_name="Statistics", index=False)

            # Autofit colunas
            for sheet_name, sheet_df in [("Pores", df), ("Statistics", stats_df)]:
                if stats_df is None and sheet_name == "Statistics":
                    continue
                if sheet_df is None or sheet_df.empty:
                    continue
                ws = writer.sheets[sheet_name]
                for col_cells in ws.iter_cols():
                    max_len = max(
                        len(str(cell.value)) if cell.value else 0
                        for cell in col_cells
                    )
                    ws.column_dimensions[col_cells[0].column_letter].width = min(
                        max_len + 4, 30
                    )

        logger.info(f"Excel salvo: {path}")
        return path

    def _export_json(self, df: pd.DataFrame, stem: str) -> Path:
        path = self.output_dir / f"{stem}.json"
        # Converte numpy types para Python nativo
        records = json.loads(
            df.to_json(orient="records", double_precision=6)
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        logger.info(f"JSON salvo: {path}")
        return path
