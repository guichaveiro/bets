"""Jogo patológico (F63.0) na atenção primária (APS) e nos CAPS, Brasil, 2018–2026: todas as análises do artigo.

Hipóteses, modelos e testes estão descritos em artigo/ANALISE.md, na mesma ordem das seções abaixo.
Só dados públicos agregados: nenhum identificador de paciente é lido (ver seção 1).

    uv run python artigo/analise.py                  # ~15 min (bootstrap com 999 réplicas)
    uv run python artigo/analise.py --B 99           # teste rápido
    uv run python artigo/analise.py --refazer-caps   # reagrega as RAAS a partir dos .dbc do DATASUS

Entradas (em data/, fora do repositório):
    data/atencao_primaria/processed/sisab_aps.parquet     relatório público do SISAB (src/saude_mental/sisab_aps.py)
    data/caps/raw/datasus/SIASUS/200801_/Dados/PS*.dbc    RAAS do SIA-PS (FTP do DATASUS; baixados se faltarem)
    data/ibge/processed/populacao_municipio.parquet       estimativas de população do IBGE
Saídas: artigo/resultados.txt e artigo/figuras/*.pdf|png
"""

# %% Configuração
import argparse
import re
import struct
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations, product
from multiprocessing import get_context
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import polars as pl
import polars.selectors as cs
import statsmodels.api as sm
from scipy import special, stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figuras"
SISAB = DATA / "atencao_primaria/processed/sisab_aps.parquet"
POP = DATA / "ibge/processed/populacao_municipio.parquet"
DBC = DATA / "caps/raw/datasus/SIASUS/200801_/Dados"
RAAS = DATA / "caps/processed/sia_raas.parquet"
RAAS_CNES = DATA / "caps/processed/sia_raas_cnes.parquet"
FTP = "ftp://ftp.datasus.gov.br/dissemin/publicos/SIASUS/200801_/Dados"

INICIO, FIM_APS, FIM_CAPS = "201801", "202609", "202606"  # RAAS chegam com atraso: jul–ago/2026 incompletos
PANDEMIA = ("202003", "202112")
KMAX, MIN_SEG, LAGS, BLOCO = 2, 12, 6, 6  # máx. de quebras; meses mínimos por segmento; defasagens HAC; bloco boot

# Grupos (CID-10 principal)
REF_CID = ["F32", "F33", "F41", "F43", "F10", "F20"]  # referência: depressão, ansiedade, estresse, álcool, esquizofrenia
CONTROLE_CID = ["F631", "F632", "F633"]  # controle negativo: piromania, cleptomania, tricotilomania
REF_SISAB = ["depressao_cid", "depressao_rec_cid", "ansiedade_cid", "estresse_cid", "alcool_cid", "esquizofrenia_cid"]
CONTROLE_SISAB = ["piromania_cid", "cleptomania_cid", "tricotilomania_cid"]
CIAP_SISAB = ["depressao_ciap", "ansiedade_ciap", "sensacao_ansiedade_ciap", "tristeza_ciap", "alcool_ciap",
              "drogas_ciap", "esquizofrenia_ciap", "suicidio_tentativa_ciap"]
UFS = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
       "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]

# %% Saída em texto
LINHAS: list[str] = []


def out(s=""):
    print(s, flush=True)
    LINHAS.append(str(s))


def secao(titulo):
    out("\n" + "=" * 100 + f"\n{titulo}\n" + "=" * 100)


def tabela(df: pl.DataFrame):
    df = df.with_columns(cs.float().map_elements(lambda x: num(x, 1), return_dtype=pl.String),
                         cs.integer().map_elements(lambda x: num(x) if abs(x) >= 1e4 else str(x),
                                                                 return_dtype=pl.String))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=60, tbl_hide_dataframe_shape=True,
                   tbl_hide_column_data_types=True, tbl_formatting="ASCII_MARKDOWN", float_precision=1):
        out(str(df))


MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def fmt_mes(m):
    return f"{MESES_PT[int(m[4:]) - 1]}/{m[:4]}"


def fmt_p(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def num(x, nd=0):
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def dec(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def pct(x):
    return f"{x:+.0f}%".replace("-", "−")


# %% 1. Dados ------------------------------------------------------------------------------------------------------
# 1a. CAPS: contagem de registros de RAAS a partir dos .dbc do SIA-PS. O .dbf tem registros de largura fixa e só os
# bytes das colunas abaixo são lidos: CNS_PAC (cartão SUS) e DTNASC nunca saem do arquivo.
COLS = ["CNES_EXEC", "DT_ATEND", "SEXOPAC", "CIDPRI", "DT_INICIO"]


def ler_colunas(dbf: Path, cols: list[str]) -> pl.DataFrame:
    b = np.fromfile(dbf, dtype=np.uint8)
    n, hl, rl = struct.unpack("<IHH", b[4:12].tobytes())
    campos, pos = {}, 1  # byte 0 de cada registro = marca de exclusão
    for i in range(32, hl - 1, 32):
        nome = b[i:i + 11].tobytes().split(b"\0")[0].decode().strip()
        campos[nome] = (pos, int(b[i + 16]))
        pos += int(b[i + 16])
    reg = b[hl:hl + n * rl].reshape(n, rl)
    reg = reg[reg[:, 0] != ord("*")]
    return pl.DataFrame({c: pl.Series(c, np.ascontiguousarray(reg[:, o:o + ln]).view(f"S{ln}").ravel()
                                      .astype(f"U{ln}")).str.strip_chars() for c, (o, ln) in
                         ((c, campos[c]) for c in cols)})


def baixar(nome: str):
    DBC.mkdir(parents=True, exist_ok=True)
    tmp = DBC / f"{nome}.part"
    for i in range(4):
        try:
            with urlopen(f"{FTP}/{nome}", timeout=90) as r, open(tmp, "wb") as f:
                while bloco := r.read(1 << 20):
                    f.write(bloco)
            tmp.rename(DBC / nome)
            return
        except OSError:
            tmp.unlink(missing_ok=True)
            if i == 3:
                raise
            time.sleep(5 * (i + 1))


def raas_arquivo(nome: str):
    """Um .dbc (UF × competência) -> (registros por mês × CID × sexo × aberta-no-mês, registros por mês × CAPS)."""
    from pyreaddbc import dbc2dbf

    if not (DBC / nome).exists():
        baixar(nome)
    with tempfile.TemporaryDirectory() as tmp:
        dbf = Path(tmp) / "a.dbf"
        dbc2dbf(str(DBC / nome), str(dbf))
        df = ler_colunas(dbf, COLS)
    c = pl.col("CIDPRI")
    df = df.with_columns(pl.col("DT_ATEND").str.slice(0, 6).alias("cmp"),
                         pl.when(c.str.starts_with("F63")).then(c).when(c.str.starts_with("F")).then(c.str.slice(0, 3))
                         .otherwise(pl.lit("outro")).alias("cid"))
    por_cid = df.group_by("cmp", "cid", pl.col("SEXOPAC").alias("sexo"),
                          (pl.col("DT_INICIO").str.slice(0, 6) == pl.col("cmp")).alias("inicio")).agg(
        pl.len().alias("registros"))
    por_cnes = pl.concat([df.filter(f).group_by("cmp", pl.col("CNES_EXEC").alias("cnes"))
                          .agg(pl.len().alias("registros")).with_columns(pl.lit(g).alias("grupo"))
                          for g, f in (("jogo", pl.col("cid") == "F630"), ("total", pl.lit(True)))])
    return por_cid, por_cnes


def refazer_caps(processos: int):
    try:
        with urlopen(FTP + "/", timeout=120) as r:
            nomes = [ln.split()[-1] for ln in r.read().decode("latin1").splitlines() if ln.strip()]
    except OSError:
        nomes = [p.name for p in DBC.glob("PS*.dbc")]
    nomes = sorted(n for n in nomes if re.fullmatch(r"PS[A-Z]{2}\d{4}\.dbc", n) and n[2:4] in UFS
                   and "20" + n[4:8] >= INICIO)
    print(f"RAAS: {len(nomes)} arquivos")
    cid, cnes = [], []
    with ProcessPoolExecutor(processos, mp_context=get_context("spawn")) as ex:
        fut = {ex.submit(raas_arquivo, n): n for n in nomes}
        for i, f in enumerate(as_completed(fut), 1):
            try:
                a, b = f.result()
                cid.append(a)
                cnes.append(b)
            except Exception as e:
                print(f"  ! {fut[f]}: {e}")
            if i % 200 == 0:
                print(f"  {i}/{len(nomes)}")
    RAAS.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(cid, how="vertical_relaxed").write_parquet(RAAS)
    pl.concat(cnes, how="vertical_relaxed").write_parquet(RAAS_CNES)


# 1b. Séries mensais nacionais
def entre(df, fim):
    return df.filter(pl.col("cmp").is_between(pl.lit(INICIO), pl.lit(fim)))


def serie_aps() -> pl.DataFrame:
    d = entre(pl.read_parquet(SISAB), FIM_APS)

    def largo(recorte, valor=None):
        x = d.filter(pl.col("recorte") == recorte)
        x = x if valor is None else x.filter(pl.col("valor") == valor)
        return x.pivot(on="grupo", index="cmp", values="atendimentos", aggregate_function="sum")

    t = largo("total").select("cmp", jogo=pl.col("jogo_cid"), referencia=pl.sum_horizontal(REF_SISAB),
                              controle=pl.sum_horizontal(CONTROLE_SISAB), trico=pl.col("tricotilomania_cid"),
                              ref_ampla=pl.sum_horizontal(REF_SISAB + CIAP_SISAB))
    for s in "FM":  # os recortes por sexo do SISAB não trazem F63.1–F63.3
        t = t.join(largo("sexo", s).select("cmp", pl.col("jogo_cid").alias(f"jogo_{s}"),
                                           pl.sum_horizontal(REF_SISAB).alias(f"referencia_{s}")), on="cmp")
    return t.fill_null(0).sort("cmp")


def serie_caps() -> pl.DataFrame:
    d = entre(pl.read_parquet(RAAS), FIM_CAPS)
    cid, sx, r, ini = pl.col("cid"), pl.col("sexo"), pl.col("registros"), pl.col("inicio")
    jogo, ref = cid == "F630", cid.is_in(REF_CID)
    t = d.group_by("cmp").agg(
        jogo=r.filter(jogo).sum(), controle=r.filter(cid.is_in(CONTROLE_CID)).sum(), trico=r.filter(cid == "F633").sum(),
        referencia=r.filter(ref).sum(), mental=r.filter(cid != "outro").sum(), total=r.sum(),
        jogo_inicio=r.filter(jogo & ini).sum(), referencia_inicio=r.filter(ref & ini).sum(),
        jogo_F=r.filter(jogo & (sx == "F")).sum(), jogo_M=r.filter(jogo & (sx == "M")).sum(),
        referencia_F=r.filter(ref & (sx == "F")).sum(), referencia_M=r.filter(ref & (sx == "M")).sum())
    c = entre(pl.read_parquet(RAAS_CNES), FIM_CAPS).filter(pl.col("grupo") == "jogo")
    c = c.group_by("cmp", "cnes").agg(r.sum()).group_by("cmp").agg(jogo_cap10=r.clip(0, 10).sum())
    return t.join(c, on="cmp", how="left").fill_null(0).sort("cmp")


def populacao() -> pl.DataFrame:
    """Brasil por ano; 2022–2023 (sem estimativa, ano do censo) interpolados; anos após o último repetem o último."""
    p = pl.read_parquet(POP).group_by("ano").agg(pl.col("populacao").sum())
    anos = pl.DataFrame({"ano": list(range(2018, 2027))}, schema={"ano": pl.Int32})
    return (anos.join(p.with_columns(pl.col("ano").cast(pl.Int32)), on="ano", how="left").sort("ano")
            .with_columns(pl.col("populacao").cast(pl.Float64).interpolate().forward_fill()))


# %% 2. Regressão segmentada (joinpoint) -----------------------------------------------------------------------------
#   g(E[y_t]) = offset_t + const + 11 dummies de mês + pandemia + β·t + Σ_k δ_k (t − τ_k)+
# NB2 com link log (offset = log do denominador) ou binomial com link logit (y de n). Segmentos contínuos.

def mes_mais(m0: str, k) -> str:
    a, m = divmod(int(m0[:4]) * 12 + int(m0[4:]) - 1 + int(k), 12)
    return f"{a}{m + 1:02d}"


def tempo(meses):
    m0 = int(meses[0][:4]) * 12 + int(meses[0][4:])
    return np.array([int(m[:4]) * 12 + int(m[4:]) - m0 for m in meses], float)


def base(meses, pandemia=True):
    """Intercepto, dummies de mês, indicador de pandemia (se houver meses de pandemia na série) e t (último)."""
    mes, t = np.array([int(m[4:]) for m in meses]), tempo(meses)
    cols, nomes = [np.ones(len(meses))] + [(mes == k) * 1.0 for k in range(2, 13)], ["const"] + [f"mes{k}" for k in range(2, 13)]
    if pandemia:
        cols.append(np.array([PANDEMIA[0] <= m <= PANDEMIA[1] for m in meses], float))
        nomes.append("pandemia")
    return np.column_stack(cols + [t]), nomes + ["t"], t


def hinges(t, taus):
    return np.column_stack([np.maximum(t - tau, 0.0) for tau in taus]) if len(taus) else np.empty((len(t), 0))


# IRLS mínimo (só para a busca das quebras e o bootstrap, que exigem milhares de ajustes; o ajuste reportado
# é o do statsmodels, em `ajustar`)
def _medias(X, beta, off, fam, n, alpha):
    eta = np.clip(off + X @ beta, -50, 50)
    if fam == "nb":
        mu = np.exp(eta)
        return eta, mu, mu / (1 + alpha * mu)
    p = np.clip(special.expit(eta), 1e-12, 1 - 1e-12)
    return eta, n * p, np.maximum(n * p * (1 - p), 1e-10)


def _desvio(y, mu, fam, n=None, alpha=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        if fam == "nb":
            return 2 * np.sum(t1 - (y + 1 / alpha) * np.log((1 + alpha * y) / (1 + alpha * mu)))
        return 2 * np.sum(t1 + np.where(n - y > 0, (n - y) * np.log((n - y) / (n - mu)), 0.0))


def _irls(X, y, off, fam, n=None, alpha=0.0, it=100, tol=1e-8):
    p0 = np.log(np.maximum(y, 0.5)) if fam == "nb" else special.logit((y + 0.5) / (n + 1.0))
    beta = np.linalg.lstsq(X, p0 - off, rcond=None)[0]
    eta, mu, w = _medias(X, beta, off, fam, n, alpha)
    dev = _desvio(y, mu, fam, n, alpha)
    for _ in range(it):
        z = eta - off + (y - mu) / (mu if fam == "nb" else w)
        new = np.linalg.lstsq(X * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None)[0]
        for _ in range(30):  # reduz o passo se o desvio piorar
            eta_n, mu_n, w_n = _medias(X, new, off, fam, n, alpha)
            dev_n = _desvio(y, mu_n, fam, n, alpha)
            if np.isfinite(dev_n) and dev_n <= dev + 1e-10:
                break
            new = (new + beta) / 2
        conv = abs(dev - dev_n) < tol * (abs(dev_n) + 0.1)
        beta, eta, mu, w, dev = new, eta_n, mu_n, w_n, dev_n
        if conv:
            break
    return beta, mu


def buscar(y, X0, t, k, fam, off, n, alpha):
    """Posições das k quebras que minimizam o desvio (busca exaustiva; ≥ MIN_SEG meses entre quebras e pontas)."""
    if k == 0:
        return ()
    nt = int(t.max()) + 1

    def ok(c):
        return c[0] >= MIN_SEG and c[-1] <= nt - MIN_SEG and all(b - a >= MIN_SEG for a, b in zip(c, c[1:]))

    def desvio(c):
        return _desvio(y, _irls(np.column_stack([X0, hinges(t, c)]), y, off, fam, n, alpha)[1], fam, n, alpha)

    passo = 1 if k <= 2 else 3  # k = 3: grade a cada 3 meses e refinamento local
    melhor = min((c for c in combinations(range(MIN_SEG, nt - MIN_SEG + 1, passo), k) if ok(c)), key=desvio)
    if passo > 1:
        melhor = min((c for c in product(*[range(b - 2, b + 3) for b in melhor]) if ok(c)), key=desvio)
    return tuple(melhor)


@dataclass
class Ajuste:
    fam: str
    meses: list
    taus: tuple
    beta: np.ndarray
    cov: np.ndarray  # HAC (Newey-West)
    nomes: list
    mu: np.ndarray
    alpha: float
    phi: float
    desvio: float
    crit: float  # BIC (NB) ou QBIC (binomial, preenchido em `selecionar`)
    y: np.ndarray
    n: np.ndarray | None
    off: np.ndarray
    X: np.ndarray
    pandemia: bool
    boot: dict = field(default_factory=dict)

    @property
    def k(self):
        return len(self.taus)

    def contraste(self, s):
        """Vetor c tal que c·β = inclinação mensal do segmento s (β_t + Σ_{j<s} δ_j)."""
        c, it = np.zeros(len(self.beta)), self.nomes.index("t")
        c[it:it + s + 1] = 1
        return c

    def segmentos(self):
        m0 = self.meses[0]
        q = [mes_mais(m0, tau) for tau in self.taus]
        return list(zip([m0] + q, q + [self.meses[-1]]))


def ajustar(y, meses, taus, fam, off=None, n=None, lags=LAGS, pandemia=True) -> Ajuste:
    off = np.zeros(len(y)) if off is None else off
    X0, nomes, t = base(meses, pandemia)
    X = np.column_stack([X0, hinges(t, taus)])
    nomes = nomes + [f"delta{j + 1}" for j in range(len(taus))]
    hac = dict(cov_type="HAC", cov_kwds={"maxlags": lags})
    if fam == "nb":
        b0, _ = _irls(X, y, off, "nb", alpha=0.05)
        r = sm.NegativeBinomial(y, X, offset=off).fit(start_params=np.r_[b0, 0.05], disp=0, maxiter=500, **hac)
        beta, alpha = r.params[:-1], r.params[-1]
        mu = np.exp(off + X @ beta)
        dev = _desvio(y, mu, "nb", alpha=alpha)
        crit, phi = -2 * r.llf + (X.shape[1] + 1 + len(taus)) * np.log(len(y)), 1.0  # + alpha + posições
        cov = r.cov_params()[:-1, :-1]
    else:
        r = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit(**hac)
        beta, mu, alpha = r.params, n * r.mu, 0.0
        dev = r.deviance
        phi = float(np.sum((y - mu) ** 2 / (mu * (1 - mu / n))) / (len(y) - X.shape[1]))
        crit, cov = np.nan, r.cov_params()
    return Ajuste(fam, meses, tuple(taus), beta, cov, nomes, mu, alpha, phi, dev, crit, y, n, off, X, pandemia)


def selecionar(y, meses, fam, off=None, n=None, kmax=KMAX, k_fixo=None, pandemia=True, lags=LAGS):
    """Ajusta K = 0..kmax quebras e escolhe o menor K com critério a menos de 2 pontos do mínimo."""
    y = np.asarray(y, float)
    off = np.zeros(len(y)) if off is None else off
    X0, _, t = base(meses, pandemia)
    alpha0 = None
    if fam == "nb":  # alpha de trabalho na busca: modelo flexível (quebras a cada 12 meses)
        alpha0 = ajustar(y, meses, tuple(range(MIN_SEG, int(t.max()) - MIN_SEG + 1, 12)), fam, off, pandemia=pandemia).alpha
    todos = {}
    for k in (range(kmax + 1) if k_fixo is None else [k_fixo]):
        taus = buscar(y, X0, t, k, fam, off, n, alpha0)
        a = ajustar(y, meses, taus, fam, off, n, lags, pandemia)
        if fam == "nb" and k:  # refaz a busca com o alpha do próprio modelo
            taus2 = buscar(y, X0, t, k, fam, off, n, a.alpha)
            a = a if taus2 == taus else ajustar(y, meses, taus2, fam, off, n, lags, pandemia)
        todos[k] = a
    if fam == "bin":  # QBIC com phi do modelo mais flexível
        phi = max(todos[max(todos)].phi, 1.0)
        for k, a in todos.items():
            a.crit = a.desvio / phi + (a.X.shape[1] + k) * np.log(len(y))
    cmin = min(a.crit for a in todos.values())
    return min((a for a in todos.values() if a.crit <= cmin + 2), key=lambda a: a.k), todos


def bootstrap(aj: Ajuste, B: int, processos: int, seed=1):
    """Bootstrap por blocos móveis (6 meses) dos resíduos de Pearson; refaz a busca das quebras em cada réplica."""
    partes = [(aj, len(p), seed * 1000 + i) for i, p in enumerate(np.array_split(np.arange(B), processos)) if len(p)]
    with ProcessPoolExecutor(len(partes), mp_context=get_context("fork")) as ex:
        res = list(ex.map(_boot, partes))
    aj.boot = {"taus": np.vstack([r[0] for r in res]), "incl": np.vstack([r[1] for r in res])}


def _boot(args):
    aj, B, seed = args
    rng = np.random.default_rng(seed)
    nt = len(aj.y)
    sd = np.sqrt(aj.mu + aj.alpha * aj.mu ** 2) if aj.fam == "nb" else np.sqrt(aj.mu * (1 - aj.mu / aj.n))
    r = (aj.y - aj.mu) / sd
    r -= r.mean()
    X0, _, t = base(aj.meses, aj.pandemia)
    taus_b, incl_b = [], []
    for _ in range(B):
        ini = rng.integers(0, nt - BLOCO + 1, int(np.ceil(nt / BLOCO)))
        ys = np.clip(np.round(aj.mu + sd * np.concatenate([r[i:i + BLOCO] for i in ini])[:nt]), 0,
                     None if aj.fam == "nb" else aj.n)
        taus = buscar(ys, X0, t, aj.k, aj.fam, aj.off, aj.n, aj.alpha)
        beta, _ = _irls(np.column_stack([X0, hinges(t, taus)]), ys, aj.off, aj.fam, aj.n, aj.alpha)
        taus_b.append(taus)
        incl_b.append([aj.contraste(s) @ beta for s in range(aj.k + 1)])
    return np.array(taus_b, float).reshape(B, aj.k), np.array(incl_b).reshape(B, aj.k + 1)


# Resumos
def vpa(inclinacao_mensal):
    """Variação percentual anual: 100·(exp(12·inclinação) − 1)."""
    return 100 * (np.exp(12 * np.asarray(inclinacao_mensal)) - 1)


def ic_vpa(est, var):
    se = np.sqrt(var)
    return vpa(est), vpa(est - 1.96 * se), vpa(est + 1.96 * se), 2 * stats.norm.sf(abs(est / se))


def vpam(aj: Ajuste, ini=None, fim=None):
    """VPA média (AAPC) entre dois meses: média das inclinações mensais, ponderada pelos meses de cada segmento."""
    t = tempo([aj.meses[0], ini or aj.meses[0], fim or aj.meses[-1]])
    i = np.arange(t[1], t[2])
    c = np.mean([aj.contraste(int(np.sum(np.array(aj.taus) <= j))) for j in i], axis=0)
    res = ic_vpa(c @ aj.beta, c @ aj.cov @ c)
    if aj.boot:
        b = [np.mean(inc[np.searchsorted(np.asarray(tb), i + 0.5)]) for tb, inc in zip(aj.boot["taus"], aj.boot["incl"])]
        res += tuple(vpa(np.percentile(b, [2.5, 97.5])))
    return res


def ljung_box(aj: Ajuste, lags=12):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    sd = np.sqrt(aj.mu + aj.alpha * aj.mu ** 2) if aj.fam == "nb" else np.sqrt(aj.mu * (1 - aj.mu / aj.n))
    return float(acorr_ljungbox((aj.y - aj.mu) / sd, lags=[lags])["lb_pvalue"].iloc[0])


def ic_quebra(aj: Ajuste, j=-1):
    q = np.percentile(aj.boot["taus"][:, j], [2.5, 97.5])
    return mes_mais(aj.meses[0], np.floor(q[0])), mes_mais(aj.meses[0], np.ceil(q[1]))


def relatar(rotulo, aj: Ajuste, todos=None):
    """Bloco de texto de um modelo: seleção de K, segmentos (VPA), mudanças de inclinação, quebras e VPAM."""
    disp = f"alpha = {dec(aj.alpha, 3)}" if aj.fam == "nb" else f"phi = {dec(aj.phi)}"
    out(f"\n{rotulo}\n  [{'binomial negativa' if aj.fam == 'nb' else 'binomial (logit)'}; {disp}; "
        f"Ljung-Box(12) dos resíduos p = {fmt_p(ljung_box(aj))}]")
    if todos:
        for k, a in todos.items():
            q = ", ".join(fmt_mes(mes_mais(a.meses[0], tau)) for tau in a.taus) or "—"
            out(f"  K = {k}: {'QBIC' if a.fam == 'bin' else 'BIC'} = {num(a.crit, 1):>9}   quebras: {q}"
                + ("   <- escolhido" if k == aj.k else ""))
    for s, (ini, fim) in enumerate(aj.segmentos()):
        c = aj.contraste(s)
        e, li, ls, p = ic_vpa(c @ aj.beta, c @ aj.cov @ c)
        linha = f"  segmento {s + 1}: {fmt_mes(ini)}–{fmt_mes(fim)}   VPA {pct(e)} (IC95% HAC {pct(li)}; {pct(ls)}; p = {fmt_p(p)})"
        if aj.boot:
            bl, bh = vpa(np.percentile(aj.boot["incl"][:, s], [2.5, 97.5]))
            linha += f"   [IC95% bootstrap {pct(bl)}; {pct(bh)}]"
        out(linha)
    it = aj.nomes.index("t")
    for j, tau in enumerate(aj.taus):
        d, v = aj.beta[it + 1 + j], aj.cov[it + 1 + j, it + 1 + j]
        linha = f"  quebra {j + 1}: {fmt_mes(mes_mais(aj.meses[0], tau))}"
        if aj.boot:
            a, b = ic_quebra(aj, j)
            linha += f" (IC95% bootstrap {fmt_mes(a)}–{fmt_mes(b)})"
        out(linha + f"; mudança de inclinação z = {dec(d / np.sqrt(v))}, p = {fmt_p(2 * stats.norm.sf(abs(d / np.sqrt(v))))}")
    r = vpam(aj)
    linha = f"  VPAM {fmt_mes(aj.meses[0])}–{fmt_mes(aj.meses[-1])}: {pct(r[0])} (IC95% HAC {pct(r[1])}; {pct(r[2])})"
    out(linha + (f" [bootstrap {pct(r[4])}; {pct(r[5])}]" if aj.boot else ""))


# %% 3–8. Análises ---------------------------------------------------------------------------------------------------
ROTULO = {
    "aps_jogo_ref": "APS · F63.0 / referência", "caps_jogo_ref": "CAPS · F63.0 / referência",
    "aps_controle_ref": "APS · F63.1–F63.3 / referência", "caps_controle_ref": "CAPS · F63.1–F63.3 / referência",
    "aps_jogo_controle": "APS · F63.0 vs F63.1–F63.3", "caps_jogo_controle": "CAPS · F63.0 vs F63.1–F63.3",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--processos", type=int, default=20)
    ap.add_argument("--refazer-caps", action="store_true")
    x = ap.parse_args()
    if x.refazer_caps or not RAAS.exists():
        refazer_caps(x.processos)

    a, c = serie_aps(), serie_caps()
    S = {"aps": a, "caps": c}
    v = lambda df, col: df[col].to_numpy().astype(float)  # noqa: E731
    M = {}
    for f, df in S.items():
        meses = df["cmp"].to_list()
        M[f"{f}_jogo_ref"] = (meses, dict(y=v(df, "jogo"), fam="nb", off=np.log(v(df, "referencia"))))
        M[f"{f}_controle_ref"] = (meses, dict(y=v(df, "controle"), fam="nb", off=np.log(v(df, "referencia"))))
        M[f"{f}_jogo_controle"] = (meses, dict(y=v(df, "jogo"), fam="bin", n=v(df, "jogo") + v(df, "controle")))

    out(f"Jogo patológico (F63.0) na APS e nos CAPS, Brasil — gerado por artigo/analise.py em {date.today():%d/%m/%Y}")
    out(f"APS: SISAB, {fmt_mes(INICIO)}–{fmt_mes(FIM_APS)} ({a.height} meses). "
        f"CAPS: RAAS/SIA-PS, {fmt_mes(INICIO)}–{fmt_mes(FIM_CAPS)} ({c.height} meses). Bootstrap: {x.B} réplicas.")

    # ---------------------------------------------------------------- 3. Descritiva
    secao("3. DESCRITIVA (registros por ano)")
    pop = populacao()
    for f, df in S.items():
        cols = ["jogo", "controle", "referencia", "jogo_F", "jogo_M"]
        an = (df.with_columns(ano=pl.col("cmp").str.slice(0, 4).cast(pl.Int32)).group_by("ano")
              .agg(*[pl.col(k).sum() for k in cols], meses=pl.len()).join(pop, on="ano").sort("ano"))
        out(f"\n{f.upper()}")
        tabela(an.select(
            "ano", "meses", pl.col("jogo").alias("F63.0"), pl.col("controle").alias("F63.1–3"),
            pl.col("referencia").alias("referência"),
            (1e4 * pl.col("jogo") / pl.col("referencia")).alias("F63.0/10 mil ref"),
            (1e4 * pl.col("controle") / pl.col("referencia")).alias("F63.1–3/10 mil ref"),
            (1e6 * pl.col("jogo") / pl.col("populacao") * 12 / pl.col("meses")).alias("F63.0/milhão hab·ano"),
            (100 * pl.col("jogo_F") / (pl.col("jogo_F") + pl.col("jogo_M"))).alias("% mulheres F63.0")))

    # ---------------------------------------------------------------- 4. H1: efeito principal
    secao("4. H1 — O REGISTRO DE JOGO PATOLÓGICO CRESCEU ALÉM DO CRESCIMENTO DO SISTEMA?")
    out("\n4a. Crescimento do sistema (denominadores, contagens sem offset; NB, K por BIC)")
    sistema = {"APS · referência (F32, F33, F41, F43, F10, F20)": (a, "referencia"),
               "APS · referência ampliada (CID + CIAP de saúde mental)": (a, "ref_ampla"),
               "CAPS · referência (F32, F33, F41, F43, F10, F20)": (c, "referencia"),
               "CAPS · todas as RAAS": (c, "total")}
    for rot, (df, col) in sistema.items():
        relatar(rot, selecionar(v(df, col), df["cmp"].to_list(), "nb")[0])

    out("\n4b. Modelo principal: F63.0 com offset log(referência) — tendência da taxa de F63.0 por registro de "
        "referência, já descontado o crescimento do sistema")
    A, TODOS = {}, {}
    for nome in ("aps_jogo_ref", "caps_jogo_ref"):
        meses, kw = M[nome]
        A[nome], TODOS[nome] = selecionar(meses=meses, **kw)
        bootstrap(A[nome], x.B, x.processos)
        relatar(ROTULO[nome], A[nome], TODOS[nome])

    # ---------------------------------------------------------------- 5. Especificidade
    secao("5. ESPECIFICIDADE — CONTROLE NEGATIVO (F63.1–F63.3: piromania, cleptomania, tricotilomania)")
    out("\n5a. Controle com o mesmo modelo do H1 (esperado: sem alta)")
    for nome in ("aps_controle_ref", "caps_controle_ref"):
        meses, kw = M[nome]
        A[nome], TODOS[nome] = selecionar(meses=meses, **kw)
        relatar(ROTULO[nome], A[nome], TODOS[nome])
    out("\n5b. Jogo contra controle: binomial F63.0/(F63.0 + F63.1–3). A inclinação é a diferença entre as taxas de "
        "crescimento do jogo e do controle (VPA = crescimento anual da razão F63.0 : controle)")
    for nome in ("aps_jogo_controle", "caps_jogo_controle"):
        meses, kw = M[nome]
        A[nome], TODOS[nome] = selecionar(meses=meses, **kw)
        bootstrap(A[nome], x.B, x.processos)
        relatar(ROTULO[nome], A[nome], TODOS[nome])

    # ---------------------------------------------------------------- 6. Robustez
    secao("6. ROBUSTEZ — VPA DO SEGMENTO FINAL EM CADA CENÁRIO (IC95% HAC)")
    sens = []

    def registrar(nome, cenario, aj):
        c_ = aj.contraste(aj.k)
        e, li, ls, _ = ic_vpa(c_ @ aj.beta, c_ @ aj.cov @ c_)
        sens.append({"modelo": nome, "cenario": cenario, "K": aj.k,
                     "quebra_final": mes_mais(aj.meses[0], aj.taus[-1]) if aj.k else "", "vpa": float(e),
                     "li": float(li), "ls": float(ls)})

    for nome in ("aps_jogo_ref", "caps_jogo_ref", "aps_jogo_controle", "caps_jogo_controle"):
        meses, kw = M[nome]
        df = S[nome.split("_")[0]]
        aj = A[nome]
        registrar(nome, "principal", aj)
        registrar(nome, "até 3 quebras", selecionar(meses=meses, kmax=3, **kw)[0])
        registrar(nome, "1 quebra (fixo)", selecionar(meses=meses, k_fixo=1, **kw)[0])
        for cen, keep, pand in (("sem meses da pandemia", [not (PANDEMIA[0] <= m <= PANDEMIA[1]) for m in meses], False),
                                ("até dez/2025", [m <= "202512" for m in meses], True)):
            keep = np.array(keep)
            kw2 = {k: (val[keep] if isinstance(val, np.ndarray) else val) for k, val in kw.items()}
            registrar(nome, cen, selecionar(meses=[m for m, k in zip(meses, keep) if k], pandemia=pand, **kw2)[0])
        registrar(nome, "HAC com 12 defasagens", ajustar(aj.y, meses, aj.taus, aj.fam, aj.off, aj.n, lags=12))
        if nome == "aps_jogo_ref":
            registrar(nome, "referência ampliada (CID + CIAP)",
                      selecionar(meses=meses, y=kw["y"], fam="nb", off=np.log(v(df, "ref_ampla")))[0])
        if nome == "caps_jogo_ref":
            registrar(nome, "referência = capítulo F inteiro",
                      selecionar(meses=meses, y=kw["y"], fam="nb", off=np.log(v(df, "mental")))[0])
            registrar(nome, "no máx. 10 registros por CAPS-mês",
                      selecionar(meses=meses, y=v(df, "jogo_cap10"), fam="nb", off=kw["off"])[0])
            registrar(nome, "só RAAS abertas no mês (casos novos)",
                      selecionar(meses=meses, y=v(df, "jogo_inicio"), fam="nb", off=np.log(v(df, "referencia_inicio")))[0])
        if nome.endswith("_controle"):
            registrar(nome, "controle = só F63.3",
                      selecionar(meses=meses, y=kw["y"], fam="bin", n=kw["y"] + v(df, "trico"))[0])
    sens = pl.DataFrame(sens)
    tabela(sens.with_columns(pl.col("modelo").replace(ROTULO),
                             pl.col("quebra_final").map_elements(lambda m: fmt_mes(m) if m else "—", return_dtype=pl.String),
                             pl.col("vpa", "li", "ls").round(0).cast(pl.Int64)))

    # ---------------------------------------------------------------- 7. Objetivo secundário: onde começou
    secao("7. SECUNDÁRIO — A ALTA COMEÇOU NA APS ANTES DO CAPS? (início do segmento final; bootstrap)")
    for par, rot in (("jogo_ref", "F63.0 / referência"), ("jogo_controle", "F63.0 vs controle")):
        qa, qc = A[f"aps_{par}"], A[f"caps_{par}"]
        desl = tempo([qa.meses[0], qc.meses[0]])[1]  # as duas séries começam em jan/2018: desl = 0
        da, dc = qa.boot["taus"][:, -1], qc.boot["taus"][:, -1] + desl
        dif = dc - da[np.random.default_rng(11).permutation(len(da))]  # réplicas independentes
        est = qc.taus[-1] + desl - qa.taus[-1]
        out(f"\n{rot}")
        out(f"  APS:  {fmt_mes(mes_mais(qa.meses[0], qa.taus[-1]))} (IC95% {' – '.join(map(fmt_mes, ic_quebra(qa)))})")
        out(f"  CAPS: {fmt_mes(mes_mais(qc.meses[0], qc.taus[-1]))} (IC95% {' – '.join(map(fmt_mes, ic_quebra(qc)))})")
        out(f"  defasagem CAPS − APS: {dec(est, 0)} meses (IC95% bootstrap {dec(np.percentile(dif, 2.5), 0)} a "
            f"{dec(np.percentile(dif, 97.5), 0)}); CAPS depois da APS em {100 * np.mean(dif > 0):.0f}% das réplicas")

    # ---------------------------------------------------------------- 8. Objetivo secundário: feminização
    secao("8. SECUNDÁRIO — FEMINIZAÇÃO: A PROPORÇÃO DE MULHERES CRESCE MAIS NO JOGO DO QUE NA REFERÊNCIA? (2022–2026)")
    out("Logística empilhada (mulheres/total; séries jogo e referência), dummies de mês, tendência própria de cada "
        "grupo; EP de Driscoll-Kraay (escores somados por mês, Newey-West com 6 defasagens).")
    for f, df in S.items():
        d = df.filter(pl.col("cmp") >= "202201")
        X0, _, t = base(d["cmp"].to_list(), pandemia=False)
        nt, g = d.height, np.r_[np.zeros(d.height), np.ones(d.height)]
        ta = np.r_[t, t] / 12  # anos
        X = np.column_stack([np.vstack([X0[:, :-1]] * 2), ta, g, g * ta])
        y = np.r_[v(d, "referencia_F"), v(d, "jogo_F")]
        n = y + np.r_[v(d, "referencia_M"), v(d, "jogo_M")]
        r = sm.GLM(np.column_stack([y, n - y]), X, family=sm.families.Binomial()).fit(
            cov_type="hac-groupsum", cov_kwds={"time": np.r_[np.arange(nt), np.arange(nt)], "maxlags": LAGS})
        b, V = r.params, r.cov_params()
        out(f"\n{f.upper()}")
        for rot, cv in (("OR por ano, referência", [-3]), ("OR por ano, jogo", [-3, -1]),
                        ("razão de OR jogo/referência (teste)", [-1])):
            e, se = b[cv].sum(), np.sqrt(V[np.ix_(cv, cv)].sum())
            out(f"  {rot:<38} {dec(np.exp(e), 2)} (IC95% {dec(np.exp(e - 1.96 * se), 2)}–{dec(np.exp(e + 1.96 * se), 2)}); "
                f"p = {fmt_p(2 * stats.norm.sf(abs(e / se)))}")
        an = d.with_columns(ano=pl.col("cmp").str.slice(0, 4)).group_by("ano").agg(
            pl.col("jogo_F", "jogo_M", "referencia_F", "referencia_M").sum()).sort("ano")
        out("  % mulheres por ano (jogo | referência): " + "  ".join(
            f"{r_['ano']}: {100 * r_['jogo_F'] / (r_['jogo_F'] + r_['jogo_M']):.0f} | "
            f"{100 * r_['referencia_F'] / (r_['referencia_F'] + r_['referencia_M']):.0f}" for r_ in an.iter_rows(named=True)))

    (OUT / "resultados.txt").write_text("\n".join(LINHAS) + "\n", encoding="utf-8")
    figuras(A, S, sens)
    print(f"\n✓ {OUT / 'resultados.txt'} e {FIG}/")


# %% 9. Figuras ------------------------------------------------------------------------------------------------------
AZUL, LARANJA, TINTA, TINTA2, GRADE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
MARCOS = {"201812": "Lei 13.756", "202312": "Lei 14.790", "202501": "Mercado regulado"}


def figuras(A, S, sens):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from statsmodels.stats.proportion import proportion_confint

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42})
    FIG.mkdir(exist_ok=True)
    d = lambda m: date(int(m[:4]), int(m[4:]), 15)  # noqa: E731
    fmt_y = plt.FuncFormatter(lambda val, _: f"{val:g}".replace(".", ","))

    def eixo_tempo(ax, tit):
        ax.set_title(tit, loc="left", fontsize=9, color=TINTA, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(fmt_y)
        ax.axvspan(d(PANDEMIA[0]), d(PANDEMIA[1]), color="#efeeea", zorder=0, lw=0)

    def salvar(fig, nome):
        fig.savefig(FIG / f"{nome}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"{nome}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def tendencia(aj):  # preditor linear sem sazonalidade (média dos efeitos de mês) e sem offset
        X, ims = aj.X.copy(), [i for i, nm in enumerate(aj.nomes) if nm.startswith("mes")]
        ef = np.r_[0, aj.beta[ims]].mean()
        X[:, ims] = 0
        return X @ aj.beta + ef

    paineis = (("aps", "A. Atenção primária (SISAB)"), ("caps", "B. CAPS (RAAS, SIA-PS)"))

    # Figura 1: volume mensal (média móvel de 3 meses) relativo a 2018 — jogo × sistema × controle
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (f, tit) in zip(axs, paineis):
        df = S[f]
        eixo_tempo(ax, tit)
        x = [d(m) for m in df["cmp"]]
        for col, cor, rot in (("jogo", AZUL, "Jogo patológico (F63.0)"), ("referencia", TINTA2, "Referência (sistema)"),
                              ("controle", LARANJA, "Controle (F63.1–F63.3)")):
            s = df[col].cast(pl.Float64)
            ax.plot(x, (s / s.head(12).mean()).rolling_mean(3, center=True, min_samples=1), color=cor, lw=1.8, label=rot)
    axs[0].set_ylabel("Registros mensais / média de 2018\n(escala log; média móvel de 3 meses)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    salvar(fig, "fig1_volume")

    # Figura 2: taxa por 10 mil registros de referência com o ajuste joinpoint e a quebra final (IC bootstrap)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, (f, tit) in zip(axs, paineis):
        aj = A[f"{f}_jogo_ref"]
        x = [d(m) for m in aj.meses]
        eixo_tempo(ax, tit)
        for nome, cor, rot, mk in ((f"{f}_jogo_ref", AZUL, "Jogo patológico (F63.0)", "o"),
                                   (f"{f}_controle_ref", LARANJA, "Controle (F63.1–F63.3)", "s")):
            b = A[nome]
            ax.scatter(x, 1e4 * b.y / np.exp(b.off), s=9, color=cor, alpha=0.45, lw=0, marker=mk, zorder=2)
            ax.plot(x, 1e4 * np.exp(tendencia(b)), color=cor, lw=2, zorder=3, label=rot)
        lo, hi = ic_quebra(aj)
        ax.axvspan(d(lo), d(hi), color=AZUL, alpha=0.10, lw=0)
        ax.axvline(d(mes_mais(aj.meses[0], aj.taus[-1])), color=AZUL, lw=1.0, ls="--")
        top = ax.get_ylim()[1]
        for i, (m, rot) in enumerate(MARCOS.items()):
            ax.axvline(d(m), color=TINTA2, lw=0.6, ls=":", zorder=1)
            ax.text(d(m), top / (1.0 if i % 2 == 0 else 1.45), " " + rot, fontsize=6, color=TINTA2, va="top",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    axs[0].set_ylabel("Registros por 10 mil registros de referência\n(escala log)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    salvar(fig, "fig2_taxa_joinpoint")

    # Figura 3: floresta da robustez
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    y, ticks, labels = 0, [], []
    for m in ("aps_jogo_ref", "caps_jogo_ref", "aps_jogo_controle", "caps_jogo_controle"):
        ax.text(-0.01, y + 0.2, ROTULO[m], transform=ax.get_yaxis_transform(), ha="right", fontweight="bold", color=TINTA)
        y -= 1
        for r in sens.filter(pl.col("modelo") == m).iter_rows(named=True):
            p = r["cenario"] == "principal"
            cor = AZUL if p else TINTA2
            ax.plot([r["li"], r["ls"]], [y, y], color=cor, lw=1.5 if p else 1)
            ax.plot(r["vpa"], y, "o" if p else "D", color=cor, ms=5 if p else 3.5)
            ticks.append(y)
            labels.append(f"{r['cenario']}  [{fmt_mes(r['quebra_final'])}]" if r["quebra_final"] else r["cenario"])
            y -= 1
        y -= 0.6
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Variação percentual anual no segmento final (IC95%)")
    ax.set_xlim(left=0)
    fig.tight_layout()
    salvar(fig, "fig3_robustez")

    # Figura 4: % de mulheres por ano (IC de Wilson)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (f, tit) in zip(axs, paineis):
        an = (S[f].with_columns(ano=pl.col("cmp").str.slice(0, 4).cast(pl.Int32)).group_by("ano")
              .agg(pl.col("jogo_F", "jogo_M", "referencia_F", "referencia_M").sum()).sort("ano"))
        for gr, cor, rot, mk, dx in (("jogo", AZUL, "Jogo patológico (F63.0)", "o", 0.08),
                                     ("referencia", TINTA2, "Referência", "s", -0.08)):
            fe, ma = an[f"{gr}_F"].to_numpy(), an[f"{gr}_M"].to_numpy()
            li, ls = proportion_confint(fe, fe + ma, method="wilson")
            p = fe / (fe + ma)
            ax.errorbar(an["ano"].to_numpy() + dx, 100 * p, yerr=[100 * (p - li), 100 * (ls - p)], color=cor,
                        marker=mk, ms=4, lw=1.5, capsize=0, label=rot)
        ax.set_title(tit, loc="left", fontsize=9, color=TINTA, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xticks(an["ano"].to_list(), [str(k) for k in an["ano"].to_list()], fontsize=7)
    axs[0].set_ylabel("% de registros em mulheres")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    salvar(fig, "fig4_mulheres")


if __name__ == "__main__":
    main()
