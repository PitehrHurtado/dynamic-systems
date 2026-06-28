"""
Simulador de lista de espera — CIRUGÍA GENERAL (SSVSA)
======================================================
Modelo de stock único (Dinámica de Sistemas). La lista de espera es un nivel
(stock) gobernado por una ecuación diferencial con un flujo de entrada y tres
flujos de salida.

ECUACIÓN DIFERENCIAL DEL STOCK
------------------------------
    dL/dt = Entrada − Atención − Fuga al privado − Egreso varios

    Entrada         = serie mensual (histórica real o ~Normal)   (ver modo_entrada)
    Atención        = min(Capacidad, L/dt)                       (limitada por capacidad)
    Fuga al privado = modelo Binomial Negativa(mediana TE)       (estocástico, ver abajo)
    Egreso varios   = egreso_varios                              (tasa constante)

    Capacidad = horas_totales · ratio_publico · ratio_atencion · pac_por_hora · n_especialistas

BUCLE SALARIAL (erosión de capacidad, lazo R1)
----------------------------------------------
Cuando la fuga al privado supera su MÁXIMO histórico (un récord), el especialista
construye capacidad privada permanente: migra horas del sector público al privado.
Esas horas NO se revierten. Si la fuga baja y vuelve a subir al mismo nivel, no
migra horas nuevas (ya las tiene asignadas). Solo un récord de fuga provoca
migración, modulada por la sensibilidad del médico a la brecha salarial.

FUGA AL PRIVADO — Regresión Binomial Negativa
---------------------------------------------
Ajustada sobre 55 meses de Cirugía General SSVSA. Datos de conteo con
sobredispersión -> Poisson inválida; se usa NB. Predictor: MEDIANA del tiempo de
espera en días. Pseudo-R² = 0.394, N = 55.
    log(mu) = beta0 + beta1 · mediana_espera_DIAS
    n_egresos ~ BinomialNegativa(media = mu, size = theta)
Se extrae UN conteo aleatorio por mes; la media crece con la espera y el ruido
replica la dispersión real. (En modo determinista se usa mu directo.)

Tiempo de espera (Ley de Little, solo reporte): TE = L / Atención
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import csv
import os
import numpy as np


# Serie real de ENTRADAS mensuales — Cirugía General SSVSA, 2021-01 a 2025-07 (55 meses)
SERIE_HISTORICA_CG = (
    219, 177, 189, 203, 215, 179, 212, 224, 266, 211, 142, 136,   # 2021
    258, 208, 255, 247, 297, 203, 279, 336, 330, 278, 241, 226,   # 2022
    218, 249, 331, 328, 364, 289, 310, 330, 255, 324, 318, 282,   # 2023
    399, 321, 344, 356, 319, 284, 295, 330, 241, 340, 329, 267,   # 2024
    337, 359, 295, 339, 309, 236, 274,                            # 2025 (ene-jul)
)


def _fecha(mes: int) -> str:
    """Etiqueta año-mes a partir del índice de mes (mes 0 = 2021-01)."""
    return "%d-%02d" % (2021 + mes // 12, mes % 12 + 1)


# =====================================================================
# ============   PARÁMETROS  (editá acá para experimentar)   ===========
# =====================================================================
@dataclass
class Parametros:
    # --- Horizonte e integración ---
    dt: float = 0.03125          # paso de integración (mes)
    t_final: float = 54.0        # nº de meses (solo modos "normal"/"constante";
    #                              "historico" usa el largo de la serie real, 55 meses)

    # --- A) Stock inicial ---
    lista_inicial: float = 683.0

    # --- B) Entrada — modos sobre la ventana 2021-2025 (NO se proyecta) ---
    #   "historico": serie REAL CG 2021-01..2025-07 -> validar contra lo observado
    #   "normal"   : sortea entradas ~ Normal(media, sd) sobre la misma ventana -> Monte Carlo
    #   "constante": entrada fija = entrada_media (útil para análisis de equilibrio)
    modo_entrada: str = "historico"
    entrada_media: float = 274.6    # media de la Normal / valor constante [entradas/mes]
    entrada_sd: float = 60.16       # sd de la Normal [entradas/mes]

    # --- C) Capacidad de atención (desagregada) ---
    horas_totales: float = 160.0   # horas/mes por especialista
    ratio_publico: float = 0.50    # fracción de horas al sector público
    ratio_atencion: float = 0.30   # fracción de esas horas a atender la lista
    pac_por_hora: float = 2.0      # pacientes atendidos por hora
    n_especialistas: float = 5.0   # número de especialistas

    # --- C bis) BUCLE SALARIAL (erosión de capacidad por récord de fuga) ---
    #   Solo un RÉCORD de fuga (delta = fuga − máximo previo > 0) migra horas:
    #     brecha_nueva = (delta / pac_por_tramo) · monto_por_tramo · sensibilidad
    #     horas_nuevas = brecha_nueva / monto_por_hora
    #     acum_horas  += horas_nuevas           (solo crece; no se revierte)
    #     horas_pub    = max(horas_pub_base − acum_horas, horas_publico_min)
    bucle_salarial: bool = True
    sensibilidad_brecha: float = 0.7            # [0-1] cuánto le importa al médico la brecha
    pacientes_por_tramo_brecha: float = 5.0     # "por cada 5 pacientes adicionales..."
    monto_brecha_por_tramo: float = 100000.0    # "...+$100.000 de brecha salarial"
    monto_brecha_por_hora: float = 100000.0     # "$100.000 de brecha = 1 hora migrada al privado"
    horas_publico_min: float = 10.0             # piso de horas públicas (h/mes)

    # --- D) Fuga al privado: Regresión Binomial Negativa (PARÁMETROS FIJOS) ---
    # Predictor: MEDIANA del tiempo de espera (días). Pseudo-R²=0.394, N=55 meses.
    beta0_priv: float = 0.497107    # intercepto (escala log)
    beta1_priv: float = 0.005595    # pendiente por día de espera (escala log)
    theta_priv: float = 2.1747      # dispersión (size)
    dias_por_mes: float = 30.44     # conversión TE meses -> días para el predictor

    # --- E) Egreso varios (administrativo) ---
    egreso_varios: float = 7.0     # pacientes/mes

    # --- Control de aleatoriedad ---
    estocastico: bool = True       # True: sortea NB cada mes | False: usa la media mu
    semilla: int | None = None     # fijar para reproducibilidad

    @property
    def capacidad(self) -> float:
        return (self.horas_totales * self.ratio_publico * self.ratio_atencion
                * self.pac_por_hora * self.n_especialistas)
# =====================================================================


def _draw_nb(rng, mu, theta):
    """Sortea un conteo Binomial Negativo con media mu y dispersión theta
    (parametrización NB2 de R: var = mu + mu^2/theta)."""
    p = theta / (theta + mu)
    return float(rng.negative_binomial(theta, p))


def _serie_entradas(p: "Parametros", rng):
    """Serie mensual de entradas según p.modo_entrada.
    'historico' usa los 55 meses reales; el resto, t_final meses."""
    if p.modo_entrada == "historico":
        return np.array(SERIE_HISTORICA_CG, dtype=float)

    n_meses = int(round(p.t_final))

    if p.modo_entrada == "constante":
        return np.full(n_meses + 1, p.entrada_media)

    if p.modo_entrada == "normal":
        if not p.estocastico:
            return np.full(n_meses + 1, p.entrada_media)
        return np.maximum(np.round(rng.normal(p.entrada_media, p.entrada_sd, n_meses + 1)), 0.0)

    raise ValueError("modo_entrada desconocido: %r" % p.modo_entrada)


def simular(p: Parametros) -> dict:
    """Integra la ecuación diferencial del stock con Euler.
    La fuga al privado se sortea una vez por mes a partir del TE de ese momento."""
    rng = np.random.default_rng(p.semilla)
    entradas = _serie_entradas(p, rng)            # serie según modo_entrada
    n_meses = len(entradas) - 1                   # el modo fija el horizonte
    n = int(round(n_meses / p.dt))
    L = p.lista_inicial
    horas_pub_base = p.horas_totales * p.ratio_publico * p.n_especialistas  # horas públicas (h/mes)
    acum_horas = 0.0        # ESTADO: horas acumuladas migradas al privado (solo crece)
    brecha = 0.0            # brecha salarial acumulada (pesos, solo reporte)
    horas_pub = horas_pub_base
    cap = horas_pub_base * p.ratio_atencion * p.pac_por_hora   # capacidad inicial

    hist = {k: [] for k in ('t', 'lista', 'entrada', 'atencion', 'fuga',
                            'egreso_varios', 'tiempo_espera', 'mu_fuga',
                            'capacidad', 'brecha', 'horas_publico',
                            'acum_horas', 'delta_fuga')}
    mes_prev = -1
    fuga_mes = 0.0
    fuga_max = 0.0        # máximo histórico de fuga (récord) alcanzado hasta ahora
    delta_fuga_mes = 0.0  # cuánto la fuga del mes supera el récord previo
    mu_mes = 0.0
    entrada_mes = entradas[0]
    for i in range(n + 1):
        t = i * p.dt
        mes = int(np.floor(t + 1e-9))

        # --- INICIO DE MES: sortear fuga y actualizar capacidad (escalón mensual) ---
        if mes != mes_prev:
            entrada_mes = entradas[min(mes, n_meses)]

            # 1. Fuga del mes: NB sobre el TE con la capacidad vigente
            atencion_m = min(cap, L / p.dt)
            te_dias = (L / max(atencion_m, 1.0)) * p.dias_por_mes
            mu_mes = np.exp(p.beta0_priv + p.beta1_priv * te_dias)
            fuga_mes = _draw_nb(rng, mu_mes, p.theta_priv) if p.estocastico else mu_mes

            # 2. Bucle salarial: solo un RÉCORD de fuga migra horas nuevas
            if p.bucle_salarial:
                delta_fuga_mes = max(fuga_mes - fuga_max, 0.0)
                brecha_nueva = (delta_fuga_mes / p.pacientes_por_tramo_brecha) * p.monto_brecha_por_tramo * p.sensibilidad_brecha
                acum_horas += brecha_nueva / p.monto_brecha_por_hora
                horas_pub = max(horas_pub_base - acum_horas, p.horas_publico_min)
                brecha = acum_horas * p.monto_brecha_por_hora  # brecha acumulada (reporte)
            else:
                delta_fuga_mes = 0.0
                brecha = 0.0
                horas_pub = horas_pub_base

            # 3. Capacidad fija para todo el mes con las horas actualizadas
            cap = horas_pub * p.ratio_atencion * p.pac_por_hora
            fuga_max = max(fuga_max, fuga_mes)   # actualizar el récord
            mes_prev = mes

        # --- Cada paso: la lista evoluciona con la capacidad fija del mes ---
        atencion = min(cap, L / p.dt)
        te_meses = L / max(atencion, 1.0)
        fuga = min(fuga_mes, max(L / p.dt - atencion, 0.0))
        egreso = min(p.egreso_varios, max(L / p.dt - atencion - fuga, 0.0))
        dL = entrada_mes - atencion - fuga - egreso

        hist['t'].append(t)
        hist['lista'].append(L)
        hist['entrada'].append(entrada_mes)
        hist['atencion'].append(atencion)
        hist['fuga'].append(fuga)
        hist['egreso_varios'].append(egreso)
        hist['tiempo_espera'].append(te_meses)
        hist['mu_fuga'].append(mu_mes)
        hist['capacidad'].append(cap)
        hist['brecha'].append(brecha)
        hist['horas_publico'].append(horas_pub)
        hist['acum_horas'].append(acum_horas)
        hist['delta_fuga'].append(delta_fuga_mes)

        # --- Integración (Euler) — acum_horas se actualiza solo al inicio de mes ---
        L = max(L + p.dt * dL, 0.0)

    return hist


def simular_replicas(p: Parametros, n_replicas: int = 200) -> dict:
    """Corre n_replicas simulaciones estocásticas y devuelve media y banda 5-95%."""
    trayectorias = []
    t_ref = None
    for r in range(n_replicas):
        h = simular(replace(p, estocastico=True, semilla=r))
        trayectorias.append(h['lista'])
        t_ref = h['t']
    M = np.array(trayectorias)
    return {'t': t_ref,
            'media': M.mean(axis=0),
            'p05': np.percentile(M, 5, axis=0),
            'p95': np.percentile(M, 95, axis=0)}


def lista_final(p: Parametros) -> float:
    """Valor de la lista en el último mes simulado (determinista por defecto)."""
    return simular(p)['lista'][-1]


def exportar_csv(hist: dict, ruta: str):
    cols = ['t', 'lista', 'entrada', 'atencion', 'fuga', 'egreso_varios',
            'tiempo_espera', 'mu_fuga', 'capacidad', 'brecha', 'horas_publico',
            'acum_horas', 'delta_fuga']
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(hist['t'])):
            w.writerow([hist[c][i] for c in cols])
    print("CSV:", ruta)


# =====================================================================
# ============   REPORTES   ============================================
# =====================================================================
def reporte_validacion(aqui: str) -> dict:
    """MODO HISTÓRICO: entradas reales 2021-2025. Para validar vs lo observado."""
    p = Parametros(modo_entrada="historico", estocastico=False)
    h = simular(p)
    print("=" * 78)
    print("MODO HISTÓRICO — entradas reales CG (validación)")
    print("Lista inicial (2021-01) = %.0f  |  Capacidad base = %.0f pac/mes"
          % (p.lista_inicial, p.capacidad))
    print("-" * 78)
    print("%9s %11s %8s %8s %10s %6s %9s" %
          ("fecha", "lista sim.", "TE(mes)", "capac.", "horas_pub", "fuga", "h_migr"))
    print("-" * 78)
    t = np.array(h['t'])
    for mes in (0, 12, 24, 36, 48, 54):
        idx = int(np.argmin(np.abs(t - mes)))
        print("%9s %11.0f %8.1f %8.0f %10.0f %6.0f %9.2f"
              % (_fecha(mes), h['lista'][idx], h['tiempo_espera'][idx],
                 h['capacidad'][idx], h['horas_publico'][idx], h['fuga'][idx],
                 h['acum_horas'][idx]))
    print("=" * 78)
    exportar_csv(h, os.path.join(aqui, "cg_historico.csv"))
    return h


def reporte_montecarlo(aqui: str, h_hist: dict):
    """MODO NORMAL: misma ventana, entradas estocásticas (banda 5-95%)."""
    print("\nMODO NORMAL — misma ventana 2021-2025, entradas ~ Normal(274.6, 60.16)")
    print("200 réplicas; el histórico real debería caer dentro de la banda")
    ens = simular_replicas(Parametros(modo_entrada="normal", t_final=54.0), n_replicas=200)
    t = np.array(ens['t'])
    print("%9s %12s %12s %12s" % ("fecha", "media", "p05", "p95"))
    for mes in (0, 12, 24, 36, 48, 54):
        idx = int(np.argmin(np.abs(t - mes)))
        print("%9s %12.0f %12.0f %12.0f"
              % (_fecha(mes), ens['media'][idx], ens['p05'][idx], ens['p95'][idx]))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.fill_between(ens['t'], ens['p05'], ens['p95'], alpha=0.25, color="steelblue",
                        label="modo normal — banda 5-95% (Monte Carlo)")
        ax.plot(ens['t'], ens['media'], color="steelblue", label="modo normal — media")
        ax.plot(h_hist['t'], h_hist['lista'], color="black", lw=2,
                label="modo histórico (entradas reales)")
        ax.set_xlabel("mes desde 2021-01"); ax.set_ylabel("pacientes en lista")
        ax.set_title("Lista de espera CG 2021-2025 — histórico vs. Monte Carlo Normal")
        ax.legend(); fig.tight_layout()
        ruta = os.path.join(aqui, "cg_historico_vs_normal.png")
        fig.savefig(ruta, dpi=110); print("Gráfico:", ruta)
    except Exception as e:
        print("(sin gráfico:", e, ")")


# =====================================================================
# ============   BATERÍA DE EXPERIMENTOS   =============================
# =====================================================================
def _barrer(param: str, valores, base: Parametros) -> None:
    """Barrido univariado: imprime lista final para cada valor de un parámetro."""
    print("  %-26s -> lista final (jul-2025)" % param)
    for v in valores:
        Lf = lista_final(replace(base, **{param: v}))
        print("    %-22s = %-8s  lista = %6.0f" % (param, v, Lf))


def bateria_experimentos():
    """Battería de experimentos de análisis del sistema (determinista, modo histórico)."""
    base = Parametros(modo_entrada="historico", estocastico=False)
    print("\n" + "#" * 78)
    print("# BATERÍA DE EXPERIMENTOS — análisis del sistema (modo histórico determinista)")
    print("#" * 78)

    # E1 — Aislar el efecto del bucle de erosión (R1)
    print("\n[E1] Efecto del BUCLE de erosión salarial (ON vs OFF)")
    print("    bucle OFF  -> lista = %6.0f" % lista_final(replace(base, bucle_salarial=False)))
    print("    bucle ON   -> lista = %6.0f" % lista_final(replace(base, bucle_salarial=True)))

    # E2 — Sensibilidad al egreso administrativo (palanca de calibración)
    print("\n[E2] Sensibilidad a EGRESO VARIOS (depuración administrativa)")
    _barrer("egreso_varios", (7, 12, 15, 19, 22, 25), base)

    # E3 — Sensibilidad a la capacidad (ratio_atencion)
    print("\n[E3] Sensibilidad a la CAPACIDAD (fracción de horas a la lista)")
    _barrer("ratio_atencion", (0.25, 0.30, 0.35, 0.40, 0.45), base)

    # E4 — Intensidad del bucle: sensibilidad del médico a la brecha
    print("\n[E4] Intensidad del BUCLE — sensibilidad del médico a la brecha")
    _barrer("sensibilidad_brecha", (0.0, 0.3, 0.7, 1.0), base)

    # E5 — Umbral de reacción del médico (pacientes por tramo de brecha)
    print("\n[E5] UMBRAL de reacción — pacientes por tramo de brecha")
    _barrer("pacientes_por_tramo_brecha", (3, 5, 10, 20, 40), base)

    # E6 — Intervención de política: aumentar dotación de especialistas
    print("\n[E6] INTERVENCIÓN — aumentar dotación de especialistas")
    _barrer("n_especialistas", (5, 6, 7, 8), base)

    # E7 — Condiciones extremas (test de robustez estructural, Barlas)
    print("\n[E7] CONDICIONES EXTREMAS (test de robustez estructural, Barlas)")
    print("    entrada = 0 (modo constante)       -> lista = %6.0f  (esperado: se vacía a 0)"
          % lista_final(replace(base, modo_entrada="constante", entrada_media=0.0)))
    print("    capacidad ~0 (ratio_atencion=0.001) -> lista = %6.0f  (NO explota: ver nota)"
          % lista_final(replace(base, ratio_atencion=0.001)))
    print("    NOTA: con capacidad ~0 el TE -> ∞ y la fuga NB exp(beta1·TE) drena la lista:")
    print("          la fuga al privado es una VÁLVULA DE ESCAPE exponencial (lazo balanceador B)")
    print("          que impide el crecimiento ilimitado. Es comportamiento estructural válido.")
    print("#" * 78)


if __name__ == "__main__":
    aqui = os.path.dirname(os.path.abspath(__file__))
    h_hist = reporte_validacion(aqui)
    reporte_montecarlo(aqui, h_hist)
    bateria_experimentos()
