"""Layer 0 materials and per-block physics.

Each tissue is mapped to a real material stack. Property values are typical literature /
datasheet values (sources in REFS, keyed by the tag in each entry). Where a tissue is a
composite (several materials in one block), block properties are computed by simple
rule-of-mixtures from the volume fractions given here; those fractions and the block size are
DESIGN ASSUMPTIONS, not measurements, and are listed as such.

Physics checks (evaluated on a grown pattern; they do not feed back into the genome rules):
  1. steady-state heat conduction: core + sensors (+ Joule heat) -> scaffold -> radiator ->
     environment (radiation from every block face with its emissivity + optional convection)
  2. power delivery: conduit blocks host local flow-cell stacks that hold the bus voltage;
     power reaches consumers through the electrically conductive block network (nodal DC solve)
  3. load / strain: (a) cantilever bending under an n*g inertial load with the body held at its
     posterior edge, (b) thermal-expansion mismatch stress at every interface between
     different materials, compared to the weaker material's strength and to the latch strength.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import genome as gn

SIGMA_SB = 5.670374e-8

# ------------------------------------------------------------------ pure materials (SI units)
# rho kg/m3, k W/mK, E Pa, strength Pa (tensile/flexural, weakest relevant), cte 1/K,
# sigma_e S/m (electrical), eps (IR emissivity of an exposed face), Tmax degC (service limit)
MAT = {
    "graphene_epoxy": dict(name="graphene-nanoplatelet / epoxy composite (25 vol% GNP)",
                           rho=1450, k=5.1, E=5e9, strength=60e6, cte=40e-6, sigma_e=10.0,
                           eps=0.85, Tmax=120, ref="shahil2012, epoxy_hb"),
    "heat_pipe": dict(name="micro heat-pipe channel (water/copper-wick, effective)",
                      rho=3000, k=1000.0, E=110e9, strength=200e6, cte=17e-6, sigma_e=0.0,  # wall dielectric-isolated (assumption)
                      eps=0.1, Tmax=150, ref="faghri2012"),
    "cu_trace": dict(name="copper power/ground traces (flex-PCB style, embedded)", rho=8960, k=400.0,
                     E=117e9, strength=210e6, cte=16.5e-6, sigma_e=5.8e7, eps=0.05, Tmax=200, ref="cu_hb"),
    "dlc": dict(name="diamond-like carbon (ta-C) coating", rho=3100, k=3.5, E=500e9,
                strength=1.0e9, cte=1.0e-6, sigma_e=1e-6, eps=0.5, Tmax=400, ref="robertson2002, shamsa2006"),
    "silica_fiber": dict(name="silica optical fibre", rho=2200, k=1.38, E=72e9, strength=1.0e9,
                         cte=0.55e-6, sigma_e=1e-12, eps=0.9, Tmax=500, ref="silica_hb"),
    "vrfb_electrolyte": dict(name="vanadium redox electrolyte (1.6 M V in H2SO4)", rho=1350,
                             k=0.6, E=0.0, strength=0.0, cte=0.0, sigma_e=30.0, eps=0.9, Tmax=40,
                             visc=5e-3, energy_Wh_per_L=25.0, ref="skyllas2011"),
    "pgs": dict(name="pyrolytic graphite sheet (in-plane)", rho=1000, k=1000.0, E=20e9,
                strength=20e6, cte=-1e-6, sigma_e=1e5, eps=0.85, Tmax=400, ref="pgs_ds"),
    "si": dict(name="crystalline silicon", rho=2330, k=148.0, E=130e9, strength=300e6,
               cte=2.6e-6, sigma_e=1e2, eps=0.6, Tmax=85, ref="si_hb"),
    "ingaas": dict(name="In0.53Ga0.47As (SWIR detector)", rho=5500, k=5.0, E=80e9, strength=100e6,
                   cte=5.7e-6, sigma_e=1e2, eps=0.6, Tmax=70, ref="adachi"),
    "ge": dict(name="germanium window (LWIR optic)", rho=5323, k=60.0, E=103e9, strength=93e6,
               cte=6.1e-6, sigma_e=2.0, eps=0.1, Tmax=100, ref="ge_znse_ds"),
    "znse": dict(name="zinc selenide window (LWIR optic)", rho=5270, k=18.0, E=67e9, strength=55e6,
                 cte=7.1e-6, sigma_e=1e-10, eps=0.1, Tmax=250, ref="ge_znse_ds"),
    "vox_bolometer": dict(name="VOx microbolometer array (uncooled)", rho=4500, k=5.0, E=100e9,
                          strength=100e6, cte=5e-6, sigma_e=1e2, eps=0.9, Tmax=80, ref="flir_lepton"),
}

REFS = {
    "shahil2012": "K.M.F. Shahil, A.A. Balandin, 'Graphene-multilayer graphene nanocomposites as highly efficient thermal interface materials', Nano Lett. 12, 861 (2012): k up to ~5.1 W/mK at high GNP loading.",
    "epoxy_hb": "Typical filled-epoxy composite handbook values (E 3-8 GPa, tensile 50-80 MPa, CTE 30-60 ppm/K); electrical conductivity of GNP/epoxy above percolation ~1-100 S/m (assumed 10 S/m).",
    "faghri2012": "A. Faghri, 'Review and advances in heat pipe science and technology', J. Heat Transfer 134, 123001 (2012): effective conductivity of heat pipes/vapour chambers 10^3-10^5 W/mK (conservative 1000 W/mK used).",
    "robertson2002": "J. Robertson, 'Diamond-like amorphous carbon', Mater. Sci. Eng. R 37, 129 (2002): ta-C density ~3.1 g/cm3, Young's modulus up to ~760 GPa, hardness up to ~80 GPa.",
    "shamsa2006": "M. Shamsa et al., 'Thermal conductivity of diamond-like carbon films', Appl. Phys. Lett. 89, 161921 (2006): 0.2-3.5 W/mK.",
    "silica_hb": "Fused silica handbook values: density 2.20 g/cm3, k 1.38 W/mK, E 72 GPa, CTE 0.55 ppm/K; coated telecom fibre proof-tested to ~0.7 GPa, pristine ~5 GPa (1 GPa used).",
    "skyllas2011": "M. Skyllas-Kazacos et al., 'Progress in flow battery research and development', J. Electrochem. Soc. 158, R55 (2011): V electrolyte energy density ~25-35 Wh/L; V(V) precipitates above ~40 degC.",
    "pgs_ds": "Panasonic PGS graphite sheet datasheet: in-plane k 700-1950 W/mK, density 0.85-2.1 g/cm3 (1000 W/mK, 1.0 g/cm3 used).",
    "cu_hb": "Annealed copper handbook values: 8.96 g/cm3, k ~400 W/mK, resistivity 1.72e-8 Ohm m (5.8e7 S/m), E ~117 GPa, CTE 16.5 ppm/K. Not in the original concept: added because composite-only paths cannot deliver power (see power ablation).",
    "si_hb": "Silicon handbook values: k 148 W/mK, density 2.33 g/cm3, E 130-169 GPa, CTE 2.6 ppm/K; commercial-grade junction limit 85 degC.",
    "adachi": "S. Adachi, 'Properties of Semiconductor Alloys' (Wiley 2009): In0.53Ga0.47As k ~5 W/mK, density ~5.5 g/cm3.",
    "ge_znse_ds": "Optical-materials datasheets (e.g. Crystran handbook): Ge k 60 W/mK, 5.32 g/cm3, rupture ~93 MPa, CTE 6.1 ppm/K; ZnSe k 18 W/mK, 5.27 g/cm3, rupture ~55 MPa, CTE 7.1 ppm/K.",
    "flir_lepton": "Uncooled VOx microbolometer cores (e.g. FLIR Lepton datasheet): operating range about -10 to +80 degC.",
}

# ------------------------------------------------------------------ tissue -> material stack
# volume fractions per block (DESIGN ASSUMPTIONS)
CU = 0.03   # copper trace volume fraction in non-conduit blocks (set 0 for the power ablation)
TISSUE_STACK = {
    gn.SCAFFOLD: {"graphene_epoxy": 0.85 - CU, "heat_pipe": 0.15, "cu_trace": CU},
    gn.SHELL: {"graphene_epoxy": 0.75 - CU, "heat_pipe": 0.15, "dlc": 0.10, "cu_trace": CU},
    gn.CONDUIT: {"graphene_epoxy": 0.30 - CU, "heat_pipe": 0.10, "vrfb_electrolyte": 0.40, "silica_fiber": 0.20, "cu_trace": CU},  # Cu = cell current collectors
    gn.CORE: {"si": 0.80 - CU, "heat_pipe": 0.20, "cu_trace": CU},
    gn.SENSOR: {"si": 0.35, "ingaas": 0.10, "vox_bolometer": 0.10, "ge": 0.15, "znse": 0.10, "graphene_epoxy": 0.20 - CU, "cu_trace": CU},
    gn.RADIATOR: {"pgs": 0.85, "graphene_epoxy": 0.15},
}
TISSUE_ROLE = {
    gn.SCAFFOLD: "skeleton + heat path (graphene composite with micro heat-pipe channels)",
    gn.SHELL: "armour skin (graphene composite + heat pipes, ta-C DLC outer coating)",
    gn.CONDUIT: "vessels/nerves: silica optical fibre (data) + vanadium redox electrolyte channel (charge + heat)",
    gn.CORE: "fixed, air-gapped compute: silicon / silicon-photonic die with heat-pipe spreader",
    gn.SENSOR: "multi-band eye: Si (VIS/NIR), InGaAs (SWIR), VOx microbolometer behind Ge/ZnSe (LWIR)",
    gn.RADIATOR: "heat rejection: pyrolytic graphite sheet, high-emissivity face",
}


def block_props(tissue: int) -> dict:
    """Rule-of-mixtures block properties. k: arithmetic (parallel, in-plane, heat-pipe dominated);
    E: Voigt; strength/Tmax: weakest load-bearing / temperature-limited constituent."""
    st = TISSUE_STACK[tissue]
    out = dict(rho=0.0, k=0.0, E=0.0, cte=0.0, sigma_e=0.0)
    Esum = 0.0
    for m, f in st.items():
        p = MAT[m]
        out["rho"] += f * p["rho"]
        out["k"] += f * p["k"]
        out["E"] += f * p["E"]
        out["sigma_e"] += f * p["sigma_e"]
        out["cte"] += f * p["E"] * p["cte"]
        Esum += f * p["E"]
    out["cte"] /= max(Esum, 1.0)
    solids = [m for m in st if MAT[m]["strength"] > 0]
    out["strength"] = min(MAT[m]["strength"] for m in solids)
    out["Tmax"] = min(MAT[m]["Tmax"] for m in st)
    out["Tmax_limited_by"] = min(st, key=lambda m: MAT[m]["Tmax"])
    out["strength_limited_by"] = min(solids, key=lambda m: MAT[m]["strength"])
    # exposed face: the surface material
    face = {gn.SHELL: "dlc", gn.RADIATOR: "pgs", gn.SENSOR: "ge"}.get(tissue, max(st, key=st.get))
    out["eps"] = MAT[face]["eps"]
    return out


PROPS = {t: block_props(t) for t in TISSUE_STACK}


def set_copper(frac):
    """Rebuild the tissue stacks with a different copper-trace fraction (ablation helper)."""
    global CU
    for t, st in TISSUE_STACK.items():
        if "cu_trace" in st:
            filler = "graphene_epoxy" if "graphene_epoxy" in st else "si"
            st[filler] += st["cu_trace"] - frac
            st["cu_trace"] = frac
    CU = frac
    PROPS.update({t: block_props(t) for t in TISSUE_STACK})


@dataclass
class PhysConfig:
    a: float = 1e-3                 # block edge (m)  -- DESIGN ASSUMPTION (1 mm voxel, sheet 1 block thick)
    T_env: float = 25.0             # environment temperature (degC)
    h_conv: float = 10.0            # natural convection W/m2K per face (0 = vacuum)
    P_core: float = 0.30            # total core compute dissipation (W)
    P_sensor_block: float = 0.5e-3  # per sensor block (W)
    V_bus: float = 1.25             # flow-cell open-circuit-ish bus voltage held at conduit blocks (V)
    droop_max: float = 0.10         # max allowed fractional voltage droop at a consumer
    g_load: float = 100.0           # inertial load factor (handling shock), in g
    latch_strength: float = 20e6    # block-to-block latch shear/tensile strength (Pa) -- ASSUMPTION
    safety: float = 2.0             # safety factor on strengths
    T_assembly: float = 25.0        # stress-free temperature


def _neighbours(alive):
    """Right and down neighbour pairs (4-connectivity) between living blocks."""
    N = alive.shape[0]
    idx = -np.ones(alive.shape, int)
    idx[alive] = np.arange(alive.sum())
    pairs = []
    for dy, dx in ((0, 1), (1, 0)):
        a = alive[: N - dy, : N - dx] & alive[dy:, dx:]
        ys, xs = np.nonzero(a)
        pairs.append(np.stack([idx[ys, xs], idx[ys + dy, xs + dx]], 1))
    return idx, np.concatenate(pairs)


def _prop_map(pattern, key):
    m = np.zeros(pattern.shape)
    for t, p in PROPS.items():
        m[pattern == t] = p[key]
    return m


def solve_power(pattern, cfg: PhysConfig):
    """DC nodal solve. Conduit blocks are held at V_bus (local flow-cell stacks); core and sensor
    blocks draw constant power; conductance between blocks = series of the two half-blocks,
    G_half = 2 * sigma * a (cube of side a, sheet one block thick)."""
    alive = pattern > 0
    idx, pr = _neighbours(alive)
    n = int(alive.sum())
    sig = _prop_map(pattern, "sigma_e")[alive]
    gi, gj = 2 * sig[pr[:, 0]] * cfg.a, 2 * sig[pr[:, 1]] * cfg.a
    G = gi * gj / (gi + gj)
    tis = pattern[alive]
    P = np.zeros(n)
    ncore = max(int((tis == gn.CORE).sum()), 1)
    P[tis == gn.CORE] = cfg.P_core / ncore
    P[tis == gn.SENSOR] = cfg.P_sensor_block
    fixed = tis == gn.CONDUIT
    if not fixed.any():
        return dict(ok=False, reason="no conduit blocks", V=None)
    L = sp.coo_matrix((np.r_[G, G, -G, -G], (np.r_[pr[:, 0], pr[:, 1], pr[:, 0], pr[:, 1]],
                                              np.r_[pr[:, 0], pr[:, 1], pr[:, 1], pr[:, 0]])), (n, n)).tocsr()
    V = np.full(n, cfg.V_bus)
    free = ~fixed
    Lff = L[free][:, free] + sp.identity(int(free.sum())) * 1e-12
    Lfc = L[free][:, fixed]
    for _ in range(30):     # constant-power loads: I = P / V (fixed point)
        I = P[free] / np.maximum(V[free], 0.05)
        Vf = spl.spsolve(Lff.tocsc(), -I - Lfc @ V[fixed])
        if np.allclose(Vf, V[free], atol=1e-7):
            V[free] = Vf
            break
        V[free] = Vf
        if Vf.min() < 0.05 * cfg.V_bus:      # network cannot carry the load: voltage collapse
            break
    cons = (tis == gn.CORE) | (tis == gn.SENSOR)
    droop = 1 - V[cons] / cfg.V_bus
    # Joule heat per block: half of each link's dissipation to each end
    dV = V[pr[:, 0]] - V[pr[:, 1]]
    plink = G * dV ** 2
    joule = np.zeros(n)
    np.add.at(joule, pr[:, 0], plink / 2)
    np.add.at(joule, pr[:, 1], plink / 2)
    Vmap = np.full(pattern.shape, np.nan)
    Vmap[alive] = V
    Jmap = np.zeros(pattern.shape)
    Jmap[alive] = joule
    Ptot = float(P.sum())
    # electrolyte flow needed to supply Ptot from the conduit stacks (eta 0.8) and its pressure drop
    e = MAT["vrfb_electrolyte"]
    Q = Ptot / (0.8 * e["energy_Wh_per_L"] * 3600 * 1e3)          # m3/s (Wh/L -> J/m3)
    w = cfg.a * np.sqrt(TISSUE_STACK[gn.CONDUIT]["vrfb_electrolyte"])  # square channel side
    Lnet = float((tis == gn.CONDUIT).sum()) * cfg.a                 # total channel length (upper bound)
    dp = 28.4 * e["visc"] * Lnet * Q / w ** 4                         # laminar square duct
    e_store = float((tis == gn.CONDUIT).sum()) * cfg.a ** 3 * TISSUE_STACK[gn.CONDUIT]["vrfb_electrolyte"] * 1e3 * e["energy_Wh_per_L"]
    if V.min() < 0.05 * cfg.V_bus:           # collapsed: report, and do not feed absurd Joule heat on
        joule[:] = 0
        Jmap[:] = 0
    return dict(ok=bool(droop.max() <= cfg.droop_max), collapsed=bool(V.min() < 0.05 * cfg.V_bus), V=Vmap, joule=Jmap, P_load=Ptot,
                P_joule=float(joule.sum()), max_droop=float(droop.max()), mean_droop=float(droop.mean()),
                flow_uL_s=Q * 1e9, dp_Pa=dp, pump_W=dp * Q, onboard_energy_Wh=e_store,
                onboard_runtime_s=e_store * 3600 / max(Ptot, 1e-12))


def solve_thermal(pattern, cfg: PhysConfig, extra_heat=None):
    """Steady 2D conduction (sheet one block thick) with radiation + convection from both faces of
    every block. Radiation is linearised and iterated."""
    alive = pattern > 0
    idx, pr = _neighbours(alive)
    n = int(alive.sum())
    k = _prop_map(pattern, "k")[alive]
    eps = _prop_map(pattern, "eps")[alive]
    gi, gj = 2 * k[pr[:, 0]] * cfg.a, 2 * k[pr[:, 1]] * cfg.a
    G = gi * gj / (gi + gj)
    L = sp.coo_matrix((np.r_[G, G, -G, -G], (np.r_[pr[:, 0], pr[:, 1], pr[:, 0], pr[:, 1]],
                                              np.r_[pr[:, 0], pr[:, 1], pr[:, 1], pr[:, 0]])), (n, n)).tocsr()
    tis = pattern[alive]
    Q = np.zeros(n)
    ncore = max(int((tis == gn.CORE).sum()), 1)
    Q[tis == gn.CORE] = cfg.P_core / ncore
    Q[tis == gn.SENSOR] = cfg.P_sensor_block
    if extra_heat is not None:
        Q += extra_heat[alive]
    A = 2 * cfg.a ** 2                      # both faces
    # open edges of the outline also exchange (side faces)
    nb = np.zeros(pattern.shape, int)
    a8 = alive.astype(int)
    nb[1:, :] += a8[:-1, :]; nb[:-1, :] += a8[1:, :]; nb[:, 1:] += a8[:, :-1]; nb[:, :-1] += a8[:, 1:]
    A = A + (4 - nb[alive]) * cfg.a ** 2
    Te = cfg.T_env + 273.15
    T = np.full(n, Te + 10)
    for _ in range(60):
        hr = eps * SIGMA_SB * (T ** 2 + Te ** 2) * (T + Te)
        hc = (hr + cfg.h_conv) * A
        M = L + sp.diags(hc)
        Tn = Te + spl.spsolve(M.tocsc(), Q)
        if np.max(np.abs(Tn - T)) < 1e-4:
            T = Tn
            break
        T = Tn
    Tmap = np.full(pattern.shape, np.nan)
    Tmap[alive] = T - 273.15
    Tmax_allowed = _prop_map(pattern, "Tmax")
    viol = alive & (Tmap > Tmax_allowed)
    q_rad = float((eps * SIGMA_SB * (T ** 4 - Te ** 4) * A).sum())
    out = dict(T=Tmap, Tmax=float(np.nanmax(Tmap)), Q_in=float(Q.sum()), Q_rad=q_rad,
               Q_conv=float((cfg.h_conv * A * (T - Te)).sum()), n_violations=int(viol.sum()),
               ok=not viol.any(), violations={})
    for t in TISSUE_STACK:
        m = pattern == t
        if m.any():
            out.setdefault("per_tissue", {})[gn.TISSUE_NAMES[t]] = dict(
                Tmax=float(np.nanmax(Tmap[m])), limit=PROPS[t]["Tmax"], limited_by=PROPS[t]["Tmax_limited_by"],
                margin=float(PROPS[t]["Tmax"] - np.nanmax(Tmap[m])))
            if (viol & m).any():
                out["violations"][gn.TISSUE_NAMES[t]] = int((viol & m).sum())
    return out


def solve_mechanics(pattern, cfg: PhysConfig, Tmap=None):
    """(a) inertial bending: body clamped at its posterior (low-x) edge, load n*g transverse (in the
    sheet plane, along y). Each column is a beam section of the living blocks in it.
    (b) thermal mismatch stress at interfaces: sigma = dalpha * dT * E1E2/(E1+E2)."""
    alive = pattern > 0
    a = cfg.a
    rho = _prop_map(pattern, "rho")
    E = _prop_map(pattern, "E")
    S = _prop_map(pattern, "strength")
    cols = np.nonzero(alive.any(0))[0]
    x0 = cols.min()
    mass_col = (rho * alive).sum(0) * a ** 3
    xs = np.arange(pattern.shape[1])
    res = []
    worst_bend = 0.0
    bend_util = 0.0
    for c in cols:
        distal = xs > c
        Mmt = float((mass_col[distal] * cfg.g_load * 9.81 * (xs[distal] - c) * a).sum())
        ys = np.nonzero(alive[:, c])[0]
        yc = ys.mean()
        I = float(((ys - yc) * a) ** 2 @ np.ones(len(ys)) * a * a + len(ys) * a ** 4 / 12)  # thickness a
        cmax = (np.abs(ys - yc).max() + 0.5) * a
        sig = Mmt * cmax / I if I > 0 else 0.0
        smin = min(S[ys, c].min(), cfg.latch_strength)
        worst_bend = max(worst_bend, sig)
        bend_util = max(bend_util, sig * cfg.safety / smin)
    # thermal mismatch
    th_util, th_worst, strain_max = 0.0, 0.0, 0.0
    if Tmap is not None:
        cte = _prop_map(pattern, "cte")
        idx, pr = _neighbours(alive)
        yy, xx = np.nonzero(alive)
        p1, p2 = (yy[pr[:, 0]], xx[pr[:, 0]]), (yy[pr[:, 1]], xx[pr[:, 1]])
        dT = (Tmap[p1] + Tmap[p2]) / 2 - cfg.T_assembly
        E1, E2 = E[p1], E[p2]
        Eeff = E1 * E2 / np.maximum(E1 + E2, 1)
        sig = np.abs(cte[p1] - cte[p2]) * np.abs(dT) * Eeff
        lim = np.minimum(np.minimum(S[p1], S[p2]), cfg.latch_strength)
        util = sig * cfg.safety / lim
        th_util, th_worst = float(util.max()), float(sig.max())
        strain_max = float((np.abs(cte[p1] - cte[p2]) * np.abs(dT)).max())
    return dict(bend_stress_max_Pa=worst_bend, bend_utilisation=bend_util,
                thermal_stress_max_Pa=th_worst, thermal_utilisation=th_util,
                mismatch_strain_max=strain_max, ok=bool(max(bend_util, th_util) <= 1.0),
                mass_g=float((rho * alive).sum() * a ** 3 * 1e3))


def evaluate(pattern, cfg: PhysConfig | None = None):
    cfg = cfg or PhysConfig()
    pw = solve_power(pattern, cfg)
    th = solve_thermal(pattern, cfg, extra_heat=pw.get("joule"))
    me = solve_mechanics(pattern, cfg, th["T"])
    return dict(power=pw, thermal=th, mech=me, ok=bool(pw["ok"] and th["ok"] and me["ok"]))


def max_core_power(pattern, cfg: PhysConfig | None = None, lo=0.0, hi=5.0):
    """Largest total core dissipation for which every block stays within its temperature limit."""
    import copy
    cfg = copy.copy(cfg or PhysConfig())
    for _ in range(30):
        mid = (lo + hi) / 2
        cfg.P_core = mid
        th = solve_thermal(pattern, cfg)
        if th["ok"]:
            lo = mid
        else:
            hi = mid
    return lo


def summary(ev):
    pw, th, me = ev["power"], ev["thermal"], ev["mech"]
    return dict(ok=ev["ok"], T_max_C=round(th["Tmax"], 2), thermal_ok=th["ok"], thermal_violations=th["violations"],
                per_tissue_T={k: round(v["Tmax"], 2) for k, v in th.get("per_tissue", {}).items()},
                heat_in_W=round(th["Q_in"], 4), radiated_W=round(th["Q_rad"], 4), convected_W=round(th["Q_conv"], 4),
                power_ok=pw["ok"], max_droop=round(pw["max_droop"], 4), joule_W=round(pw["P_joule"], 5),
                load_W=round(pw["P_load"], 4), flow_uL_s=round(pw["flow_uL_s"], 3), dp_Pa=round(pw["dp_Pa"], 2),
                onboard_runtime_s=round(pw["onboard_runtime_s"], 1),
                mech_ok=me["ok"], bend_util=round(me["bend_utilisation"], 5), thermal_stress_MPa=round(me["thermal_stress_max_Pa"] / 1e6, 3),
                thermal_stress_util=round(me["thermal_utilisation"], 4), mass_g=round(me["mass_g"], 3))
