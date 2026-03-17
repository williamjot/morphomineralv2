# Segmentação de Poros — EDS / CBS / QEMSCAN

> **Plano de Trabalho v1.0 — 2025**  
> Análise morfométrica com Ilastik 1.4 + PARTISAN v2.0  
> _Documento para fornecedor de desenvolvimento_

---

## Sumário

1. [Objetivo do Projeto](#1-objetivo-do-projeto)
2. [Escopo de Trabalho](#2-escopo-de-trabalho)
3. [⚠️ Restrição Crítica — Ilastik e conda](#3-restrição-técnica-crítica--ilastik-e-conda)
4. [Arquitetura do Software](#4-arquitetura-do-software)
5. [Inputs e Outputs do Sistema](#5-inputs-e-outputs-do-sistema)
6. [Stack Técnico](#6-stack-técnico)
7. [Entregáveis](#7-entregáveis)
8. [Plano de Sprints](#8-plano-de-sprints)
9. [Critérios de Aceite Final](#9-critérios-de-aceite-final)
10. [Material Fornecido pelo Cliente](#10-material-fornecido-pelo-cliente)
11. [Parâmetros Configuráveis](#11-parâmetros-configuráveis-configyaml)
12. [Comandos de Uso da CLI](#12-comandos-de-uso-da-cli)
13. [Plano de Testes](#13-plano-de-testes)
14. [Glossário](#14-glossário)
15. [Observações Finais](#15-observações-finais)

---

## 1. Objetivo do Projeto

Desenvolver um software Python de linha de comando (CLI) capaz de:

- Carregar um modelo de segmentação de pixels previamente treinado no Ilastik 1.4.1 (arquivo `.ilp`);
- Aplicar esse modelo sobre imagens EDS, CBS ou QEMSCAN (grayscale ou multicanal) para classificar cada pixel em uma de três classes: **poro**, **partícula mineral** ou **matriz**;
- Extrair a máscara binária de poros e identificar cada poro individualmente;
- Calcular as **41 métricas morfométricas do PARTISAN v2.0** para cada poro segmentado;
- Exportar os resultados em CSV, Excel e PDF.

> O código base já foi desenvolvido e está disponível para ser entregue ao fornecedor como referência técnica e ponto de partida.

---

## 2. Escopo de Trabalho

### ✅ Dentro do escopo

- Leitura e parsing do arquivo `.ilp` (HDF5) para extrair metadados do modelo treinado;
- Carregamento de imagens nos formatos TIFF, PNG e BMP (single-channel e multi-channel);
- Normalização de imagens para `float32 [0, 1]` antes da inferência;
- Execução da segmentação via `ilastik.experimental.api` (modo headless, sem interface gráfica);
- Binarização do mapa de probabilidades com threshold de Otsu (padrão), fixo ou adaptativo;
- Pós-processamento morfológico: opening, closing e fill holes;
- Identificação de poros individuais via componentes conectados com filtro de área mínima;
- Cálculo das 41 métricas morfométricas do PARTISAN v2.0 para cada poro;
- Exportação de resultados em CSV, Excel (`.xlsx`) e JSON;
- Geração de visualizações em PNG: overlay binário, labels individuais, mapa de probabilidade, histogramas e scatter plot;
- Geração automática de relatório PDF com resumo da análise;
- Arquivo de configuração YAML (`config.yaml`) para todos os parâmetros ajustáveis;
- Suporte a processamento em batch (múltiplas imagens em um diretório);
- Testes unitários para os módulos críticos.

### ❌ Fora do escopo

- Treinamento de novos modelos no Ilastik (o modelo `.ilp` já existe e foi fornecido pelo cliente);
- Interface gráfica (GUI) — o software é exclusivamente CLI;
- Integração com banco de dados ou sistemas externos;
- Análise estatística avançada além das métricas do PARTISAN;
- Suporte a formatos proprietários QEMSCAN além de TIFF/PNG exportados;
- Qualquer customização do código do PARTISAN v2.0 (fornecido pelo cliente).

---

## 3. Restrição Técnica Crítica — Ilastik e conda

> [!WARNING]
> **O Ilastik NÃO pode ser instalado via `pip`.** O pacote oficial exige `conda` (canal `ilastik-forge`). Esta é uma restrição imposta pelos próprios desenvolvedores do Ilastik e não tem solução via pip. O ambiente de desenvolvimento e de produção **DEVEM** usar conda.

### Comando de instalação do ambiente

```bash
conda create -n pore_seg python=3.10 -c ilastik-forge -c conda-forge ilastik
conda activate pore_seg
pip install -r requirements.txt
```

Todos os demais pacotes (`scikit-image`, `pandas`, `matplotlib`, `fpdf2`, etc.) são instalados normalmente via `pip` após a ativação do ambiente conda.

O fornecedor deve garantir que o ambiente de entrega usa conda. Qualquer tentativa de usar apenas pip resultará em falha ao importar `ilastik.experimental.api`.

---

## 4. Arquitetura do Software

O software é organizado em módulos independentes e testados separadamente. O fluxo de dados é estritamente linear (sem loops de retorno), o que facilita depuração e substituição de componentes.

### Fluxo de dados completo

```
[.ilp]  ──>  ilp_reader.py  ──>  Metadados do modelo (classes, índice poro)
                                           |
[imagem TIFF/PNG]  ──>  loader.py  ──>  normalizer.py  ──>  validator.py
                                           |
                                segmentor.py  (Ilastik API)
                                           |
                           Mapa de probabilidades  (H × W × C)
                                           |
                             thresholder.py  ──>  morphology.py
                                           |
                                Máscara binária de poros
                                           |
                                      labeler.py
                                           |
                           N poros individuais  (PoreRegion)
                                           |
                           partisan/runner.py  ──>  [PARTISAN v2.0]
                                           |
                       DataFrame  (N linhas × 41+ colunas)
                                           |
                 exporter.py   /   visualizer.py   /   reporter.py
                       |                  |                  |
                 CSV / Excel          PNG (5 fig.)          PDF
```

### Estrutura de diretórios

```
pore_segmentation/
├── main.py               # Entry point CLI (--ilp, --image, --dir, --inspect)
├── config.yaml           # Todos os parâmetros configurados aqui
├── requirements.txt
│
├── core/
│   ├── ilp_reader.py     # Leitura HDF5 do .ilp
│   └── segmentor.py      # Inferência via ilastik.experimental.api
│
├── preprocessing/
│   ├── loader.py         # Carregamento TIFF/PNG/BMP
│   ├── normalizer.py     # float32 [0,1] por canal
│   └── validator.py      # Compatibilidade com o modelo
│
├── postprocessing/
│   ├── thresholder.py    # Threshold: Otsu / fixo / adaptativo
│   ├── morphology.py     # Opening, closing, fill holes
│   └── labeler.py        # Componentes conectados + filtro de área
│
├── partisan/
│   ├── partisan.py       # PARTISAN v2.0 (fornecido — NÃO modificar)
│   └── runner.py         # Loop sobre poros → DataFrame
│
├── output/
│   ├── exporter.py       # CSV, Excel, JSON
│   ├── visualizer.py     # PNG: overlay, labels, heatmap, histogramas
│   └── reporter.py       # PDF automático
│
└── tests/
    ├── test_ilp_reader.py
    ├── test_postprocessing.py
    └── test_partisan_runner.py
```

---

## 5. Inputs e Outputs do Sistema

| Tipo | Formato | Descrição |
|------|---------|-----------|
| **INPUT** | Arquivo `.ilp` | Modelo treinado no Ilastik 1.4.1 (3 classes: poro, partícula, matriz) |
| **INPUT** | Imagens (TIFF / PNG / BMP) | Grayscale ou multicanal (EDS, CBS, QEMSCAN) |
| **INPUT** | `config.yaml` | Parâmetros: threshold, morfologia, classe poro, diretório de saída |
| **OUTPUT** | CSV / Excel | Uma linha por poro; 41 métricas PARTISAN + metadados (`area_px`, `bbox`) |
| **OUTPUT** | JSON | Mesmo conteúdo do CSV em formato estruturado |
| **OUTPUT** | PNG — overlay binário | Imagem original + poros em vermelho |
| **OUTPUT** | PNG — labels individuais | Cada poro com cor única |
| **OUTPUT** | PNG — mapa de probabilidade | Heatmap da saída do Ilastik |
| **OUTPUT** | PNG — histogramas PARTISAN | Distribuição das 10 métricas principais |
| **OUTPUT** | PDF — relatório | Resumo completo: config, métricas, imagens |

---

## 6. Stack Técnico

| Componente | Biblioteca | Observação |
|-----------|-----------|-----------|
| Leitura do `.ilp` | `h5py >= 3.9` | O `.ilp` é um arquivo HDF5 |
| Segmentação (inferência) | `ilastik 1.4.1` **(conda)** | Único componente que **EXIGE** conda |
| Carregamento de imagens | `tifffile` + `imageio` + `Pillow` | Suporte TIFF, PNG, BMP |
| Processamento de arrays | `numpy >= 1.24` | |
| Morfologia binária | `scikit-image >= 0.21` + `scipy` | |
| Análise morfométrica | `PARTISAN v2.0` (fornecido) | Arquivo `partisan.py` já pronto |
| Tabelas de resultados | `pandas >= 2.0` + `openpyxl` | |
| Visualização | `matplotlib >= 3.7` | |
| Relatório PDF | `fpdf2 >= 2.7` | |
| Configuração | `PyYAML >= 6.0` | Parâmetros em `config.yaml` |
| Testes | `pytest >= 7.4` | |
| Progresso | `tqdm >= 4.66` | |

> **Python mínimo:** 3.10. **SO:** Linux (recomendado) ou macOS. Windows com WSL2 é possível mas não testado.

---

## 7. Entregáveis

| # | Entregável | Critério de Aceite | Módulo |
|---|-----------|-------------------|--------|
| 1 | Leitura do arquivo `.ilp` (metadados, classes, features) | Classes impressas corretamente; índice do poro detectado | `core/ilp_reader.py` |
| 2 | Carregamento de imagens (grayscale e multicanal) | TIFF/PNG carregados sem erro; shape e dtype corretos | `preprocessing/loader.py` |
| 3 | Normalização para `float32 [0,1]` | Saída float32 com valores em [0,1]; funciona por canal | `preprocessing/normalizer.py` |
| 4 | Segmentação via Ilastik (mapa de probabilidades) | Saída shape `(H,W,C)`; canal poro compatível com treino | `core/segmentor.py` |
| 5 | Threshold + morfologia binária | Máscara bool reproduz resultado visual do Ilastik GUI | `postprocessing/` |
| 6 | Labeling de poros individuais e filtro de área | N poros identificados; porosidade% calculada corretamente | `postprocessing/labeler.py` |
| 7 | Análise PARTISAN por poro | DataFrame com 41 métricas por poro; sem falhas no loop | `partisan/runner.py` |
| 8 | Exportação de resultados (CSV, Excel, JSON) | Arquivos gerados e abrindo corretamente no Excel | `output/exporter.py` |
| 9 | Visualizações (overlay, histogramas, scatter) | Imagens PNG salvas; overlay alinhado com imagem | `output/visualizer.py` |
| 10 | Relatório PDF automático | PDF gerado com imagens, tabelas e métricas corretas | `output/reporter.py` |

---

## 8. Plano de Sprints

> **Estimativa total: 24 dias úteis (~5 semanas).** O cronograma pressupõe dedicação parcial ao projeto. Ajuste conforme a disponibilidade da equipe.

| Sprint | Escopo de trabalho | Duração | Entregáveis |
|--------|-------------------|---------|------------|
| **Sprint 0** — Setup | Configurar ambiente conda; instalar Ilastik 1.4.1; validar `.ilp` com `ilp_reader.py`; confirmar índice da classe poro | 3 dias | #1 |
| **Sprint 1** — Inferência | Implementar `loader` + `normalizer` + `validator`; integrar `segmentor.py` com Ilastik API; testar em 1 imagem real com comparação visual ao GUI | 5 dias | #2, #3, #4 |
| **Sprint 2** — Pós-proc | Implementar threshold (Otsu prioritário); morfologia binária + fill holes; labeling com filtro de área; validar contagem de poros | 4 dias | #5, #6 |
| **Sprint 3** — PARTISAN | Integrar `runner.py` com PARTISAN; loop sobre todos os poros; coletar DataFrame; calcular estatísticas descritivas | 4 dias | #7 |
| **Sprint 4** — Output | Implementar `exporter` CSV/Excel; `visualizer` overlay + histogramas; relatório PDF automático; teste end-to-end com 3 imagens | 5 dias | #8, #9, #10 |
| **Sprint 5** — Entrega | Testes finais com imagens reais; documentação (README + config); pacote zip entregável; handoff e treinamento | 3 dias | Todos |

---

## 9. Critérios de Aceite Final

A entrega será considerada aprovada quando **todos os 8 critérios** abaixo estiverem verificados pelo cliente com as imagens reais do projeto:

| Critério | Evidência esperada |
|---------|-------------------|
| Segmentar uma imagem TIFF grayscale de ponta a ponta | CSV com métricas PARTISAN gerado sem erros |
| Segmentar uma imagem multicanal (RGB ou >3 canais) | Mesmo pipeline funciona sem alterações de código |
| Porosidade calculada bate com resultado visual do Ilastik GUI | Diferença < 2% de área relativa |
| PARTISAN retorna 41 métricas por poro | Colunas confirmadas no Excel de saída |
| Overlay PNG alinhado com a imagem original | Verificação visual pelo cliente |
| Relatório PDF gerado automaticamente | PDF abrindo com imagens e tabelas corretas |
| Processar pasta com 10+ imagens em batch | Sem crashes; log de erros por imagem |
| Testes unitários passando | `pytest -v` sem falhas nos 3 arquivos de teste |

---

## 10. Material Fornecido pelo Cliente

- `partisan.py` — PARTISAN v2.0 completo (1005 linhas, pronto para uso);
- `model.ilp` — modelo Ilastik 1.4.1 treinado com 3 imagens, 3 classes: poro, partícula, matriz;
- 3 imagens de referência usadas no treino para validação visual da segmentação;
- `pore_segmentation.zip` — código base de referência com todos os módulos já implementados e validados sintaticamente.

### Sobre o código base fornecido

O cliente já desenvolveu um código base completo (`pore_segmentation.zip`) que implementa todos os módulos listados na seção 4. O fornecedor deve:

1. Receber o zip e revisar a estrutura;
2. Configurar o ambiente conda e verificar que `ilp_reader.py` lê corretamente o `model.ilp` fornecido;
3. Validar a segmentação em pelo menos uma imagem real comparando o resultado visual com o Ilastik GUI;
4. Corrigir eventuais incompatibilidades específicas das imagens do cliente;
5. Garantir que todos os critérios de aceite da seção 9 são atendidos.

> [!CAUTION]
> **O fornecedor NÃO deve reescrever o código do PARTISAN (`partisan.py`).** Este arquivo deve ser usado exatamente como fornecido. Qualquer alteração neste arquivo invalida a comparabilidade dos resultados com a literatura científica.

---

## 11. Parâmetros Configuráveis (`config.yaml`)

Todos os parâmetros do pipeline estão centralizados em `config.yaml` e não requerem modificação de código:

```yaml
ilastik:
  project_path: model.ilp       # Caminho para o arquivo .ilp
  pore_class_index: 0           # Índice da classe poro (0 = primeira)
  n_threads: 4                  # Threads para inferência
  ram_mb: 4096                  # RAM limite para Ilastik

preprocessing:
  normalization: percentile     # percentile | minmax | none

postprocessing:
  threshold_method: otsu        # otsu | fixed | adaptive
  threshold_value: 0.5          # Usado apenas com method=fixed
  morphology_opening_radius: 2  # 0 = desativado
  morphology_closing_radius: 2
  morphology_fill_holes: true
  min_pore_area_px: 50          # Poros menores que isso são ignorados

output:
  output_dir: results
  export_formats: [csv, excel]
  generate_overlay: true
  generate_report: true
```

---

## 12. Comandos de Uso da CLI

**Inspecionar o `.ilp`** (descobrir o índice correto da classe poro — fazer isso primeiro):

```bash
python main.py --inspect model.ilp
```

**Processar uma única imagem:**

```bash
python main.py --ilp model.ilp --image imagem.tif
```

**Processar um diretório inteiro em batch:**

```bash
python main.py --ilp model.ilp --dir pasta_imagens/
```

**Forçar o índice da classe poro manualmente** (se a detecção automática falhar):

```bash
python main.py --ilp model.ilp --image img.tif --pore-index 1
```

---

## 13. Plano de Testes

O projeto inclui três arquivos de teste unitário:

- **`tests/test_ilp_reader.py`** — valida leitura do `.ilp` com arquivo HDF5 mockado; testa detecção automática da classe poro; testa força do índice manual;
- **`tests/test_postprocessing.py`** — valida threshold Otsu e fixo; testa opening/closing/fill holes; testa labeling e filtro de área;
- **`tests/test_partisan_runner.py`** — testa loop sobre poros simulados; valida que o DataFrame retorna 41 colunas do PARTISAN; testa estatísticas descritivas.

**Rodar todos os testes:**

```bash
cd pore_segmentation
pytest tests/ -v
```

---

## 14. Glossário

| Termo | Definição |
|-------|-----------|
| `.ilp` | Arquivo de projeto do Ilastik (formato HDF5 com modelo Random Forest + features + anotações) |
| **EDS** | Energy Dispersive X-ray Spectroscopy — imagem de composição química mineral |
| **CBS** | Compositional Backscattered Electron — imagem de elétrons retroespalhados |
| **QEMSCAN** | Sistema automatizado de análise mineral por SEM-EDS |
| **Pixel classification** | Workflow do Ilastik que classifica cada pixel em uma classe usando Random Forest |
| **Probability map** | Saída do Ilastik: array `(H × W × C)` com a probabilidade de cada pixel pertencer a cada classe |
| **PARTISAN v2.0** | PARTicle Shape ANalyzer — software de morfometria 2D baseado em 6 publicações científicas |
| **Morfometria** | Mensuração e análise quantitativa de formas geométricas |
| **Porosidade** | Fração percentual da área da imagem ocupada por poros |

---

## 15. Observações Finais

- O fornecedor deve confirmar o recebimento do `model.ilp` e das imagens de referência **antes de iniciar o Sprint 1**;
- O índice correto da classe poro no `model.ilp` deve ser verificado no Sprint 0 usando `python main.py --inspect model.ilp` e reportado ao cliente para confirmar antes de prosseguir;
- Qualquer divergência entre o resultado da segmentação via código e o resultado visual do Ilastik GUI deve ser reportada **imediatamente** ao cliente;
- O código do PARTISAN (`partisan.py`) **não deve ser modificado em nenhuma hipótese**;
- Toda a comunicação de progresso deve referenciar os entregáveis numerados da seção 7.

---

*Versão 1.0 — 2025 — Uso Restrito*
