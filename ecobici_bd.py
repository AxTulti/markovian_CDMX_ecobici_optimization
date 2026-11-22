"""
ecobici_bd.py - Birth-death model for a single EcoBici station.

Pipeline
--------
0. load_trips            parse and clean the trip table
1. build_stays           turn trips into "bike b was docked at station s from t1 to t2"
   occupancy             step function N(t) for one station
   build_segments        cut time into pieces where N is constant (and grid-aligned)
   annotate_events       arrivals / departures with the state just before them
2. estimate_rates        censoring-corrected MLE of lambda (arrivals) and mu (departures)
   assumption tests      dispersion (between days, within bins), clustering,
                         state dependence, bin-width selection
3. CTMC computations     generator, transient distribution, stationary law,
                         relaxation time, hitting times, optimal starting stock
4. evaluation            time-rescaling test, predictive scoring vs baselines
5. simulate_trips        synthetic data with known rates (to test the pipeline)

Units: time in hours for all rates (events per hour).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import expm

# Fixed categorical order for plots (blue, orange, aqua, yellow)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# ----------------------------------------------------------------------------
# 0. Loading and cleaning
# ----------------------------------------------------------------------------
_COLMAP = {
    "genero_usuario": "Genero_Usuario",
    "edad_usuario": "Edad_Usuario",
    "bici": "Bici",
    "ciclo_estacion_retiro": "Ciclo_Estacion_Retiro",
    "fecha_retiro": "Fecha_Retiro",
    "hora_retiro": "Hora_Retiro",
    "ciclo_estacionarribo": "Ciclo_EstacionArribo",
    "ciclo_estacion_arribo": "Ciclo_EstacionArribo",
    "fecha_arribo": "Fecha_Arribo",
    "hora_arribo": "Hora_Arribo",
}


def _parse_dt(fecha: pd.Series, hora: pd.Series) -> pd.Series:
    """Fecha = 'YYYY-MM-DD', Hora = '1900-01-01 HH:MM:SS' (dummy date).
    Uses the date from Fecha and the clock time from Hora; returns naive datetime64[ns]."""
    date = fecha.astype(str).str.strip().str[:10]                  # '2026-08-31'
    time = hora.astype(str).str.strip().str.split(" ").str[-1]     # '23:54:59'
    out = pd.to_datetime(date + " " + time, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return out.astype("datetime64[ns]")


def _clean_id(x: pd.Series) -> pd.Series:
    return x.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_trips(src, dayfirst: bool = True, max_hours: float = 6.0,
               min_same_station_s: float = 60.0) -> tuple[pd.DataFrame, dict]:
    """Read one CSV, a list of CSVs, or a DataFrame. Returns (clean trips, report).

    Adds columns: from, to (station ids as str), bike, t_out, t_in, dur_min.
    """
    if isinstance(src, pd.DataFrame):
        df = src.copy()
    elif isinstance(src, (list, tuple)):
        df = pd.concat([pd.read_csv(p, dtype=str) for p in src], ignore_index=True)
    else:
        df = pd.read_csv(src, dtype=str)
    df = df.rename(columns={c: _COLMAP.get(c.strip().lower(), c) for c in df.columns})

    df["from"] = _clean_id(df["Ciclo_Estacion_Retiro"])
    df["to"] = _clean_id(df["Ciclo_EstacionArribo"])
    df["bike"] = _clean_id(df["Bici"])
    df["t_out"] = _parse_dt(df["Fecha_Retiro"], df["Hora_Retiro"])
    df["t_in"] = _parse_dt(df["Fecha_Arribo"], df["Hora_Arribo"])
    df["dur_min"] = (df["t_in"] - df["t_out"]).dt.total_seconds() / 60

    rep = {"rows_read": len(df)}
    m_nat = df["t_out"].isna() | df["t_in"].isna()
    m_neg = ~m_nat & (df["dur_min"] <= 0)
    m_long = ~m_nat & (df["dur_min"] > max_hours * 60)
    m_short = ~m_nat & (df["from"] == df["to"]) & (df["dur_min"] * 60 < min_same_station_s)
    rep.update(unparseable_dates=int(m_nat.sum()), non_positive_duration=int(m_neg.sum()),
               longer_than_max=int(m_long.sum()), same_station_too_short=int(m_short.sum()))
    df = df[~(m_nat | m_neg | m_long | m_short)].drop_duplicates(
        subset=["bike", "t_out", "t_in", "from", "to"])
    rep["rows_kept"] = len(df)
    
    for c in ("t_out", "t_in"):
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            raise TypeError(f"{c} is {df[c].dtype}, expected datetime64 - check _parse_dt")
    return df.reset_index(drop=True), rep


# ----------------------------------------------------------------------------
# 1. Occupancy reconstruction
# ----------------------------------------------------------------------------
def truck_visit_times(win_start: pd.Series, win_end: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Assign a truck-visit time to each rebalancing move at one station.

    Definitions
    -----------
    Move i is the pair w_i = (s_i, e_i):
        s_i = end of the bike's last trip (last time it was seen)
        e_i = start of its next trip, from a different station
    The truck moved the bike at some time in [s_i, e_i].

    W = {w_1, ..., w_n} is the set of moves; win_start[i] = s_i, win_end[i] = e_i.

    A visit is a subset V of W with a time tau_V lying in every window of V:
        tau_V in [max_{h in V} s_h, min_{h in V} e_h]   (must be non-empty)

    Goal: partition W into the fewest visits V_1, ..., V_m (m <= n).

    Algorithm (greedy interval stabbing, optimal for this goal)
    -----------------------------------------------------------
    U <- W  (unassigned pairs)
    while U is not empty:
        e* <- min of e_h over U
        V  <- {(s_h, e_h) in U : s_h <= e*}
            Every pair in U has e_h >= e*, so e* lies in every window of V,
            and the intersection [max_V s_h, e*] is non-empty.
        tau_V <- (max_V s_h + e*) / 2      midpoint of the intersection
        U <- U minus V

    Each pair ends up in exactly one visit.

    Invalid pairs (e_i < s_i, from overlapping trips in the data) are excluded
    from U at the start: they get time NaT and group size 0.

    Returns
    -------
    (tau, size): two Series aligned with the input index.
        tau[i]  = time of the visit containing pair i
        size[i] = number of pairs in that visit (0 if invalid)
    """
    order = np.argsort(win_end.values, kind="stable")
    s, e = win_start.values[order], win_end.values[order]
    tau = np.full(len(s), np.datetime64("NaT"), dtype="datetime64[ns]") # Contains times of alleged truck visit (interval / 2)
    size = np.zeros(len(s), int)
    unassigned = s <= e # only assing valid times (the start needs to happen before the end)
    for i in range(len(s)):
        if not unassigned[i]:
            continue
        members = unassigned & (s <= e[i]) # Filter all elements belonging to the ith elem of e
        lo = s[members].max()
        tau[members] = lo + (e[i] - lo) / 2
        size[members] = members.sum()
        unassigned &= ~members
        
    # Repack into input order
    t_out = pd.Series(pd.NaT, index=win_start.index, dtype="datetime64[ns]")
    g_out = pd.Series(0, index=win_start.index)
    t_out.iloc[order] = tau # time interval / 2 per travel (considerinf each s_i as a travel: some s_i may have same t/2 value)
    g_out.iloc[order] = size # number of elements of the "membership window" to which each s_i belongs
    return t_out, g_out


def build_stays(df: pd.DataFrame, station: str | None = None, rebal_cap_h: float = 12.0):
    """Convert trips into stays (intervals a bike is docked at a station).

    For bike b with trips sorted by time:
      - consistent:  arrives at s, next trip leaves from s  -> stay at s
      - rebalanced:  arrives at A, next leaves from B != A  -> truck moved it
        somewhere in the window [arrival at A, next pickup at B].
        If `station` is given, the move time at that station comes from
        truck_visit_times (fewest visits).  Moves that end up alone in their
        group (and all moves when station is None) use the window midpoint,
        with each side capped at rebal_cap_h (very long windows usually mean
        the bike was in the workshop, not docked).
      - head: before a bike's first trip it is assumed at its first origin
      - tail: after its last trip it is assumed at its last destination
    Returns (stays, rebalancing_moves, report).
    """
    d = df.sort_values(["bike", "t_out"]).reset_index(drop=True)
    t0, t1 = d["t_out"].min(), d["t_in"].max()
    g = d.groupby("bike", sort=False)
    nxt_out = g["t_out"].shift(-1) # This is a 1 shifted list of the times of travel begginings
    nxt_from = g["from"].shift(-1)
    first = g.cumcount() == 0
    has_next = nxt_out.notna()
    consistent = has_next & (d["to"] == nxt_from)
    rebal = has_next & ~consistent
    cap = pd.Timedelta(hours=rebal_cap_h)

    parts = [
        pd.DataFrame({"bike": d.bike[consistent], "station": d.to[consistent],
                      "start": d.t_in[consistent], "end": nxt_out[consistent], "kind": "observed"}),
    ]
    ws, we = d.t_in[rebal], nxt_out[rebal]
    mid = ws + (we - ws) / 2 # Time at which we are going to consider the relocation to happen 
                             # (Halfway trough last in and next out (s.t. id_in != id_out))
                             
    # Window [ws, we]: bike last seen at station A (ws), next seen at station B (we).
    # Default guess: truck moved it at the midpoint.
    # If half the window exceeds `cap`, count the bike at each station for only
    # `cap` hours (long windows usually mean workshop, not docked).
    end_A   = np.minimum(mid, ws + cap)   # bike counted at A during [ws, end_A)
    start_B = np.maximum(mid, we - cap)   # bike counted at B during [start_B, we)
                                         
    if station is not None:
        # Filter for relocations, cosider relocations at which the bikes were 
        # I.  Relocated to station
        # II. Relocated from station
        for side, mask in (("Relocated from Station", d.to[rebal] == station), ("Relocated to Station", nxt_from[rebal] == station)):
            if mask.any():
                tau, gsize = truck_visit_times(ws[mask], we[mask])
                grouped = gsize > 1 # Masked of the truck travels that had more than one bicicle
                idx = tau.index[grouped]
                if side == "Relocated from Station":
                    end_A.loc[idx] = tau[grouped]
                else:
                    start_B.loc[idx] = tau[grouped]
                    
    parts.append(pd.DataFrame({"bike": d.bike[rebal], "station": d.to[rebal],
                               "start": ws, "end": end_A, "kind": "rebal_origin"}))
    parts.append(pd.DataFrame({"bike": d.bike[rebal], "station": nxt_from[rebal],
                               "start": start_B, "end": we, "kind": "rebal_dest"}))
    last = ~has_next
    last_cap = pd.Timedelta(hours=rebal_cap_h)
    
    parts.append(pd.DataFrame({"bike": d.bike[last], "station": d.to[last],
                           "start": d.t_in[last],
                           "end": (d.t_in[last] + last_cap).clip(upper=t1),
                           "kind": "tail"}))
    parts.append(pd.DataFrame({"bike": d.bike[first], "station": d["from"][first],
                            "start": (d.t_out[first] - last_cap).clip(lower=t0),
                            "end": d.t_out[first], "kind": "head"}))

    stays = pd.concat(parts, ignore_index=True)

    bad = stays["end"] < stays["start"]  # overlapping trips (data errors)
    
    moves = pd.DataFrame({"bike": d.bike[rebal], "from_station": d.to[rebal],
                          "to_station": nxt_from[rebal], "window_start": d.t_in[rebal],
                          "window_end": nxt_out[rebal]})
    rep = {"stays": len(stays), "overlapping_dropped": int(bad.sum()),
           "rebalancing_moves": int(rebal.sum()),
           "share_rebalanced": float(rebal.sum() / max(has_next.sum(), 1)),
           "data_start": t0, "data_end": t1}
    stays = stays[~bad]
    if station is not None:
        stays = stays[stays.station == station]
        moves = moves[(moves.from_station == station) | (moves.to_station == station)]
    stays = stays.assign(idle_h=(stays.end - stays.start).dt.total_seconds() / 3600)
    return stays.reset_index(drop=True), moves.reset_index(drop=True), rep


def occupancy(stays: pd.DataFrame, station: str) -> pd.DataFrame:
    """Step function: N(t) = number of stays at `station` covering t.
    Returns DataFrame(t, N) where N holds from t until the next row."""
    st = stays[stays.station == station]
    ev = pd.concat([pd.DataFrame({"t": st.start, "d": 1}),
                    pd.DataFrame({"t": st.end, "d": -1})])
    ev = ev.groupby("t", as_index=False)["d"].sum().sort_values("t")
    ev["N"] = ev["d"].cumsum().astype(int)
    return ev[["t", "N"]].reset_index(drop=True)


def estimate_capacity(occ: pd.DataFrame, start=None, end=None) -> tuple[int, pd.Series]:
    """C = most common daily maximum of N(t).  A busy station hits its true
    capacity on many days; reconstruction errors only occasionally push the
    maximum above it.  Prefer the official dock count when you have it."""
    o = occ.copy()
    if start is not None:
        o = o[(o.t >= start) & (o.t < end)]
    dm = o.groupby(o.t.dt.normalize()).N.max()
    counts = dm.value_counts().sort_index()
    return int(counts.idxmax()), counts


def clip_occupancy(occ: pd.DataFrame, C: int) -> pd.DataFrame:
    return occ.assign(N=occ.N.clip(0, C))


def n_at(occ: pd.DataFrame, times) -> np.ndarray:
    """N(t) at given times (value in force at t, i.e. right-continuous)."""
    times = np.asarray(pd.to_datetime(times), dtype="datetime64[ns]")
    idx = np.searchsorted(occ.t.values, times, side="right") - 1
    return np.where(idx >= 0, occ.N.values[np.clip(idx, 0, None)], 0)


def n_before(occ: pd.DataFrame, times) -> np.ndarray:
    """N(t-) just before given times (the state an event 'sees')."""
    times = np.asarray(pd.to_datetime(times), dtype="datetime64[ns]")
    idx = np.searchsorted(occ.t.values, times, side="left") - 1
    return np.where(idx >= 0, occ.N.values[np.clip(idx, 0, None)], 0)


def daytype_of(dates, holidays=None) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates))
    hol = set(pd.to_datetime(list(holidays or [])).normalize())
    weekend = (dates.dt.dayofweek >= 5) | dates.dt.normalize().isin(hol)
    return np.where(weekend, "weekend", "weekday")


def _minute_of_day(t: pd.Series) -> np.ndarray:
    return ((t - t.dt.normalize()).dt.total_seconds() / 60).values


def build_segments(occ: pd.DataFrame, start, end, grid_minutes: int = 15, holidays=None):
    """Pieces of time on which N is constant and that never cross a grid line.
    Columns: start, end, dur_h, N, date, daytype, minute (minute of day at start)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    grid = pd.date_range(start.floor("D"), end.ceil("D"), freq=f"{grid_minutes}min")
    pts = np.concatenate([occ.t.values, grid.values, [start.to_datetime64(), end.to_datetime64()]])
    pts = np.unique(pts[(pts >= start.to_datetime64()) & (pts <= end.to_datetime64())])
    N = n_at(occ, pts)
    seg = pd.DataFrame({"start": pts[:-1], "end": pts[1:], "N": N[:-1]})
    seg["dur_h"] = (seg.end - seg.start).dt.total_seconds() / 3600
    seg["date"] = seg.start.dt.normalize()
    seg["daytype"] = daytype_of(seg.date, holidays)
    seg["minute"] = _minute_of_day(seg.start)
    seg = seg[seg.dur_h > 0].reset_index(drop=True)
    seg.attrs["grid_minutes"] = grid_minutes
    return seg


def annotate_events(times, occ: pd.DataFrame, holidays=None) -> pd.DataFrame:
    t = pd.Series(pd.to_datetime(times)).sort_values().reset_index(drop=True)
    return pd.DataFrame({"t": t, "N_before": n_before(occ, t), "date": t.dt.normalize(),
                         "daytype": daytype_of(t, holidays), "minute": _minute_of_day(t)})


def station_events(df, station, occ, start, end, holidays=None):
    """Arrivals (births) and departures (deaths) of one station inside [start, end)."""
    arr = df.loc[(df.to == station) & (df.t_in >= start) & (df.t_in < end), "t_in"]
    dep = df.loc[(df["from"] == station) & (df.t_out >= start) & (df.t_out < end), "t_out"]
    return annotate_events(arr, occ, holidays), annotate_events(dep, occ, holidays)


# ----------------------------------------------------------------------------
# 2. Rate estimation (censoring-corrected Poisson MLE)
# ----------------------------------------------------------------------------
def _avail(N, kind, C):
    """Indicator that an event of this kind can happen: Y(t)."""
    return (N < C) if kind == "arr" else (N > 0)


def garwood_ci(k, E, alpha=0.05):
    k, E = np.asarray(k, float), np.asarray(E, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(k > 0, stats.chi2.ppf(alpha / 2, 2 * k) / 2, 0.0) / E
        hi = stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2 / E
    return lo, hi


def estimate_rates(seg, arr, dep, C, bin_minutes=60, days=None) -> pd.DataFrame:
    """Per (daytype, bin): counts, exposures, corrected and naive rates (per hour).

        lam = arrivals / time with N < C        (corrected)
        mu  = departures / time with N > 0      (corrected)
        *_naive = count / total time            (ignores censoring)
    """
    if days is not None:
        days = pd.DatetimeIndex(days)
        seg, arr, dep = (x[x.date.isin(days)] for x in (seg, arr, dep))
    nb = 1440 // bin_minutes
    s = seg.assign(bin=(seg.minute // bin_minutes).astype(int))
    key = ["daytype", "bin"]
    idx = pd.MultiIndex.from_product([sorted(s.daytype.unique()), range(nb)], names=key)
    out = pd.DataFrame(index=idx)
    out["E_tot"] = s.groupby(key).dur_h.sum()
    out["E_arr"] = s[s.N < C].groupby(key).dur_h.sum()
    out["E_dep"] = s[s.N > 0].groupby(key).dur_h.sum()
    for name, ev in (("A", arr), ("D", dep)):
        e = ev.assign(bin=(ev.minute // bin_minutes).astype(int))
        out[name] = e.groupby(key).size()
    out = out.fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["lam"] = np.where(out.E_arr > 0, out.A / out.E_arr, 0.0)
        out["mu"] = np.where(out.E_dep > 0, out.D / out.E_dep, 0.0)
        out["lam_naive"] = np.where(out.E_tot > 0, out.A / out.E_tot, 0.0)
        out["mu_naive"] = np.where(out.E_tot > 0, out.D / out.E_tot, 0.0)
    out["lam_lo"], out["lam_hi"] = garwood_ci(out.A, out.E_arr)
    out["mu_lo"], out["mu_hi"] = garwood_ci(out.D, out.E_dep)
    out["frac_empty"] = 1 - out.E_dep / out.E_tot.replace(0, np.nan)
    out["frac_full"] = 1 - out.E_arr / out.E_tot.replace(0, np.nan)
    out.attrs.update(bin_minutes=bin_minutes, C=C)
    return out


def rate_lookup(rates, daytypes, minutes, col):
    bm = rates.attrs["bin_minutes"]
    bins = (np.asarray(minutes) // bm).astype(int)
    idx = pd.MultiIndex.from_arrays([np.asarray(daytypes), bins])
    return rates[col].reindex(idx).fillna(0.0).values


def chronological_split(dates, test_frac=0.2):
    days = np.array(sorted(pd.DatetimeIndex(pd.Series(dates).unique())))
    k = int(round(len(days) * (1 - test_frac)))
    return pd.DatetimeIndex(days[:k]), pd.DatetimeIndex(days[k:])


# ----------------------------------------------------------------------------
# 2b. Assumption tests
# ----------------------------------------------------------------------------
def between_day_dispersion(seg, ev, rates, kind, C, days=None, min_expected=5.0, min_avail_frac=0.5):
    """Do days of the same type share one rate?  For each (daytype, bin):
        X^2 = sum_days (k_j - r E_j)^2 / (r E_j)  ~  chi^2_{J-1}  under H0.
    phi = X^2/df (dispersion ratio, 1 under H0).
    Day-bins where the station was available less than min_avail_frac of the
    bin are dropped: their tiny expected counts make Pearson terms explode."""
    col = "lam" if kind == "arr" else "mu"
    bm = rates.attrs["bin_minutes"]
    if days is not None:
        seg, ev = seg[seg.date.isin(days)], ev[ev.date.isin(days)]
    s = seg[_avail(seg.N, kind, C)].assign(bin=lambda x: (x.minute // bm).astype(int))
    E = s.groupby(["daytype", "bin", "date"]).dur_h.sum()
    k = ev.assign(bin=(ev.minute // bm).astype(int)).groupby(["daytype", "bin", "date"]).size()
    t = pd.DataFrame({"E": E, "k": k}).fillna(0.0)
    t = t[t.E >= min_avail_frac * bm / 60]
    t["e"] = t.E * rates[col].reindex(t.index.droplevel("date")).values
    t = t[t.e > 0]
    res = t.groupby(level=["daytype", "bin"]).apply(lambda g: pd.Series({
        "days": len(g), "events": g.k.sum(), "mean_expected": g.e.mean(),
        "X2": ((g.k - g.e) ** 2 / g.e).sum()}))
    res["df"] = res.days - 1
    res = res[(res.df > 0) & (res.mean_expected * res.days >= min_expected)]
    res["phi"] = res.X2 / res.df
    res["p"] = stats.chi2.sf(res.X2, res.df)
    return res


def within_bin_dispersion(seg, ev, kind, C, bin_minutes, grid_minutes=15, days=None, min_avail_frac=0.5):
    """Inside each (day, bin), conditional on the bin total K, a Poisson process
    with constant rate spreads its events over sub-slots multinomially with
    probabilities E_u / sum(E_u).  Pearson X^2 summed over days ~ chi^2.
    grid_minutes must be a multiple of the grid used in build_segments and
    divide bin_minutes.  Sub-slots available < min_avail_frac are dropped."""
    seg_grid = seg.attrs.get("grid_minutes", grid_minutes)
    if grid_minutes % seg_grid or bin_minutes % grid_minutes or grid_minutes == bin_minutes:
        raise ValueError(f"need seg grid ({seg_grid}) | grid_minutes ({grid_minutes}) | bin_minutes ({bin_minutes}), "
                         "with grid_minutes < bin_minutes")
    if days is not None:
        seg, ev = seg[seg.date.isin(days)], ev[ev.date.isin(days)]
    s = seg[_avail(seg.N, kind, C)].assign(slot=lambda x: (x.minute // grid_minutes).astype(int))
    E = s.groupby(["daytype", "date", "slot"]).dur_h.sum()
    k = ev.assign(slot=(ev.minute // grid_minutes).astype(int)).groupby(["daytype", "date", "slot"]).size()
    t = pd.DataFrame({"E": E, "k": k}).fillna(0.0).reset_index()
    t = t[t.E >= min_avail_frac * grid_minutes / 60]
    t["bin"] = (t.slot * grid_minutes // bin_minutes).astype(int)
    grp = t.groupby(["daytype", "date", "bin"])
    t["K"] = grp.k.transform("sum")
    t["w"] = t.E / grp.E.transform("sum")
    t["m"] = grp.E.transform("size")
    t = t[(t.K > 0) & (t.m > 1)]
    t["e"] = t.K * t.w
    t["x2"] = (t.k - t.e) ** 2 / t.e
    per_day = t.groupby(["daytype", "bin", "date"]).agg(x2=("x2", "sum"), m=("m", "first"), K=("K", "first"))
    res = per_day.groupby(level=["daytype", "bin"]).agg(X2=("x2", "sum"), events=("K", "sum"),
                                                        df=("m", lambda m: (m - 1).sum()))
    res["phi"] = res.X2 / res.df
    res["p"] = stats.chi2.sf(res.X2, res.df)
    return res


def pooled(res):
    X2, df = res.X2.sum(), res.df.sum()
    return {"X2": X2, "df": int(df), "phi": X2 / df, "p": stats.chi2.sf(X2, df),
            "bins": len(res), "share_bins_p<0.05": float((res.p < 0.05).mean())}


def clustering_check(ev, rates, kind, window_s=60):
    """Batch arrivals?  Under Poisson, P(gap < w) = 1 - exp(-r w).
    Compares the observed number of gaps shorter than w with its expectation."""
    col = "lam" if kind == "arr" else "mu"
    t = ev.t.sort_values().values
    gaps = np.diff(t).astype("timedelta64[ms]").astype(float) / 1000
    r = rate_lookup(rates, ev.daytype.values[1:], ev.minute.values[1:], col)
    obs = int((gaps < window_s).sum())
    exp_ = float((1 - np.exp(-r * window_s / 3600)).sum())
    return {"window_s": window_s, "observed_short_gaps": obs, "expected_short_gaps": exp_,
            "ratio": obs / exp_ if exp_ > 0 else np.nan,
            "p_excess": float(stats.poisson.sf(obs - 1, exp_))}


def _ipf_poisson(k, E, factors, max_iter=500, tol=1e-10):
    """MLE of k ~ Poisson(E * prod_f theta_f[level]) by iterative proportional
    fitting: each factor level is rescaled so that its fitted total matches its
    observed total.  Returns fitted means and the log-likelihood."""
    k, E = np.asarray(k, float), np.asarray(E, float)
    codes = [pd.factorize(f)[0] for f in factors]
    theta = [np.ones(c.max() + 1) for c in codes]
    m = E.copy()
    for _ in range(max_iter):
        delta = 0.0
        for c, th in zip(codes, theta):
            obs = np.bincount(c, weights=k, minlength=len(th))
            fit = np.bincount(c, weights=m, minlength=len(th))
            ratio = np.where(fit > 0, obs / np.where(fit > 0, fit, 1), 1.0)
            th *= ratio
            m = m * ratio[c]
            delta = max(delta, np.abs(np.log(np.where((ratio > 0), ratio, 1))).max())
        if delta < tol:
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.sum(np.where(k > 0, k * np.log(m), 0.0) - m)
    return m, ll


def state_dependence_test(seg, ev, kind, C, bin_minutes, days=None, edges=(1, 2, 3, 5, 10), day_scale=None):
    """Does the rate depend on n beyond the boundary?  Multiplicative Poisson model:
        H0: r = a_cell [* d_date]                (cell = daytype x bin)
        H1: r = a_cell [* d_date] * b_group(n)   (group = bikes available / free docks)
    LR = 2(l1 - l0) ~ chi^2_{G-1}.  Also returns observed/expected by group under H0.

    Busy days also empty the station, so day-to-day variation can masquerade as
    state dependence.  Do NOT fix this with a free effect per date: n is driven by
    the same day's earlier events, which biases fixed-effect Poisson.  Instead pass
    day_scale (Series indexed by date, e.g. system-wide trips that day / mean),
    an outside measure of how busy the day was; exposures are multiplied by it."""
    if days is not None:
        seg, ev = seg[seg.date.isin(days)], ev[ev.date.isin(days)]
    lvl_seg = seg.N if kind == "dep" else C - seg.N
    lvl_ev = ev.N_before if kind == "dep" else C - ev.N_before
    labels = [f"{a}" if b == a + 1 else f"{a}-{b - 1}" for a, b in zip(edges[:-1], edges[1:])] + [f"{edges[-1]}+"]
    cut = lambda x: pd.cut(x, list(edges) + [np.inf], right=False, labels=labels)
    s = seg.assign(bin=(seg.minute // bin_minutes).astype(int), grp=cut(lvl_seg))
    e = ev.assign(bin=(ev.minute // bin_minutes).astype(int), grp=cut(lvl_ev))
    s, e = s[lvl_seg >= edges[0]], e[lvl_ev >= edges[0]]
    key = ["daytype", "bin", "grp", "date"]
    t = pd.DataFrame({"E": s.groupby(key, observed=True).dur_h.sum(),
                      "k": e.groupby(key, observed=True).size()}).fillna(0.0).reset_index()
    t = t[t.E > 0]
    if day_scale is not None:
        t["E"] = t.E * day_scale.reindex(t.date).fillna(1.0).values
    cell = t.daytype + "_" + t.bin.astype(str)
    f0 = [cell]
    m0, ll0 = _ipf_poisson(t.k, t.E, f0)
    m1, ll1 = _ipf_poisson(t.k, t.E, f0 + [t.grp.astype(str)])
    LR = 2 * (ll1 - ll0)
    dfree = t.grp.nunique() - 1
    t["fit0"] = m0
    oe = t.groupby("grp", observed=True).agg(events=("k", "sum"), expected_H0=("fit0", "sum"),
                                             exposure_h=("E", "sum"))
    oe["obs/exp"] = oe.events / oe.expected_H0
    return {"LR": LR, "df": dfree, "p": float(stats.chi2.sf(LR, dfree)), "table": oe}


def system_day_scale(df, holidays=None) -> pd.Series:
    """Trips in the whole system per day, divided by the mean for its day type.
    An outside proxy for how busy a day was (weather, events, holidays)."""
    n = df.groupby(df.t_out.dt.normalize()).size()
    dt = pd.Series(daytype_of(n.index, holidays), index=n.index)
    return n / n.groupby(dt).transform("mean")


def poisson_loglik(seg, ev, rates, kind, C, days=None, floor=0.01):
    """Log-likelihood of a Poisson process with intensity r(t) Y(t):
        l = sum_j log r(t_j) - integral r(t) Y(t) dt."""
    col = "lam" if kind == "arr" else "mu"
    if days is not None:
        seg, ev = seg[seg.date.isin(days)], ev[ev.date.isin(days)]
    r_ev = np.maximum(rate_lookup(rates, ev.daytype, ev.minute, col), floor)
    r_seg = np.maximum(rate_lookup(rates, seg.daytype, seg.minute, col), floor)
    integral = (r_seg * _avail(seg.N, kind, C) * seg.dur_h).sum()
    return float(np.log(r_ev).sum() - integral)


def select_bin_width(seg, arr, dep, C, train_days, widths=(15, 30, 60, 120, 180), val_frac=0.25):
    """Hold-out log-likelihood per event for several bin widths (higher is better)."""
    fit_d, val_d = chronological_split(train_days, val_frac)
    rows = []
    for w in widths:
        r = estimate_rates(seg, arr, dep, C, w, days=fit_d)
        la = poisson_loglik(seg, arr, r, "arr", C, val_d)
        ld = poisson_loglik(seg, dep, r, "dep", C, val_d)
        na, nd = arr.date.isin(val_d).sum(), dep.date.isin(val_d).sum()
        rows.append({"bin_minutes": w, "loglik_arr_per_event": la / na,
                     "loglik_dep_per_event": ld / nd, "loglik_total": la + ld})
    return pd.DataFrame(rows).set_index("bin_minutes")


# ----------------------------------------------------------------------------
# 3. CTMC computations
# ----------------------------------------------------------------------------
def generator(lam: float, mu: float, C: int) -> np.ndarray:
    """Q for lambda_n = lam 1{n<C}, mu_n = mu 1{n>0} on {0..C}."""
    Q = np.zeros((C + 1, C + 1))
    i = np.arange(C)
    Q[i, i + 1] = lam
    Q[i + 1, i] = mu
    Q[np.diag_indices(C + 1)] = -Q.sum(axis=1)
    return Q


def bin_rates(rates, daytype, lam_col="lam", mu_col="mu"):
    nb = 1440 // rates.attrs["bin_minutes"]
    r = rates.loc[daytype].reindex(range(nb)).fillna(0.0)
    return r[lam_col].values, r[mu_col].values


def step_matrices(rates, daytype, C, lam_col="lam", mu_col="mu", substeps=1):
    """P_b = exp(Q_b * Delta / substeps) for each bin b."""
    dt = rates.attrs["bin_minutes"] / 60 / substeps
    lam, mu = bin_rates(rates, daytype, lam_col, mu_col)
    return [expm(generator(l, m, C) * dt) for l, m in zip(lam, mu)]


def transition_matrix(mats, b0, b1):
    """Phi(b0 -> b1) = P_{b0} P_{b0+1} ... P_{b1-1}  (ordered: earliest first)."""
    Phi = np.eye(mats[0].shape[0])
    for b in range(b0, b1):
        Phi = Phi @ mats[b]
    return Phi


def day_profile(rates, daytype, C, p0, start_h=0.0, end_h=24.0, substeps=4, **cols):
    """p(t) on a fine grid from start_h to end_h; p0 is an int (n0) or a vector."""
    bm = rates.attrs["bin_minutes"]
    mats = step_matrices(rates, daytype, C, substeps=substeps, **cols)
    p = np.zeros(C + 1)
    if np.isscalar(p0):
        p[int(p0)] = 1.0
    else:
        p = np.asarray(p0, float)
    b0, b1 = int(start_h * 60 // bm), int(np.ceil(end_h * 60 / bm))
    ts, ps = [start_h], [p.copy()]
    for b in range(b0, b1):
        for s in range(substeps):
            p = p @ mats[b]
            ts.append((b + (s + 1) / substeps) * bm / 60)
            ps.append(p.copy())
    return np.array(ts), np.array(ps)


def stationary(lam, mu, C):
    """pi_n = (1-rho) rho^n / (1 - rho^{C+1});  uniform if rho = 1."""
    n = np.arange(C + 1)
    if mu == 0 and lam == 0:
        return np.full(C + 1, np.nan)
    if mu == 0:
        return (n == C).astype(float)
    if lam == 0:
        return (n == 0).astype(float)
    rho = lam / mu
    if abs(rho - 1) < 1e-9:
        return np.full(C + 1, 1 / (C + 1))
    w = rho ** (n - C) if rho > 1 else rho ** n  # stable for large/small rho
    return w / w.sum()


def spectral_gap(lam, mu, C):
    """Smallest nonzero eigenvalue of -Q:  lam + mu - 2 sqrt(lam mu) cos(pi/(C+1))."""
    return lam + mu - 2 * np.sqrt(lam * mu) * np.cos(np.pi / (C + 1))


def hitting_time_empty(lam, mu, C):
    """E[time to reach 0 | N(0)=n], n=0..C, constant rates (hours).
    u_m = (1/mu) sum_{k=0}^{C-m} rho^k,  T_n = sum_{m<=n} u_m."""
    if mu == 0:
        return np.r_[0.0, np.full(C, np.inf)]
    rho = lam / mu
    u = np.array([sum(rho ** k for k in range(C - m + 1)) / mu for m in range(1, C + 1)])
    return np.r_[0.0, np.cumsum(u)]


def hitting_time_full(lam, mu, C):
    """E[time to reach C | N(0)=n]: mirror image (n -> C-n, lam <-> mu)."""
    return hitting_time_empty(mu, lam, C)[::-1]


def hitting_time_linear(lam, mu, C, target=0):
    """Same quantity by solving the first-step linear system (for checking)."""
    Q = generator(lam, mu, C)
    keep = [n for n in range(C + 1) if n != target]
    A = Q[np.ix_(keep, keep)]
    T = np.linalg.solve(A, -np.ones(len(keep)))
    out = np.zeros(C + 1)
    out[keep] = T
    return out


def losses_by_start_stock(rates, daytype, C, start_h=5.0, end_h=24.0, lam_col="lam", mu_col="mu"):
    """For every starting stock n0 at start_h: expected lost pickups, lost returns,
    hours empty and hours full until end_h.  Exact integrals via Van Loan:
        expm([[Q, I], [0, 0]] * D) = [[e^{QD}, int_0^D e^{Qs} ds], [0, I]]."""
    bm = rates.attrs["bin_minutes"]
    lam, mu = bin_rates(rates, daytype, lam_col, mu_col)
    K, D = C + 1, bm / 60
    Phi = np.eye(K)
    acc = {k: np.zeros(K) for k in ("lost_pickups", "lost_returns", "hours_empty", "hours_full")}
    for b in range(int(start_h * 60 // bm), int(end_h * 60 // bm)):
        M = np.zeros((2 * K, 2 * K))
        M[:K, :K] = generator(lam[b], mu[b], C)
        M[:K, K:] = np.eye(K)
        E = expm(M * D)
        PF = Phi @ E[:K, K:]
        acc["lost_pickups"] += mu[b] * PF[:, 0]
        acc["lost_returns"] += lam[b] * PF[:, C]
        acc["hours_empty"] += PF[:, 0]
        acc["hours_full"] += PF[:, C]
        Phi = Phi @ E[:K, :K]
    out = pd.DataFrame(acc)
    out.index.name = "n0"
    out["lost_total"] = out.lost_pickups + out.lost_returns
    return out


def simulate_ctmc(rates, daytype, C, n0, start_h=0.0, end_h=24.0, rng=None):
    """Gillespie simulation with piecewise-constant rates (for Monte Carlo checks)."""
    rng = np.random.default_rng(rng)
    bm = rates.attrs["bin_minutes"]
    lam, mu = bin_rates(rates, daytype)
    t, n = start_h, n0
    ts, ns = [t], [n]
    while t < end_h:
        b = int(t * 60 // bm)
        t_edge = min((b + 1) * bm / 60, end_h)
        a_up = lam[b] if n < C else 0.0
        a_dn = mu[b] if n > 0 else 0.0
        a = a_up + a_dn
        dt = rng.exponential(1 / a) if a > 0 else np.inf
        if t + dt >= t_edge:  # memoryless: restart the clock at the bin edge
            t = t_edge
            continue
        t += dt
        n += 1 if rng.random() < a_up / a else -1
        ts.append(t)
        ns.append(n)
    return np.array(ts), np.array(ns)


# ----------------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------------
def time_rescaling(seg, ev, rates, kind, C, days):
    """Lambda(t) = int_0^t r(s) Y(s) ds.  If the model is right, the rescaled gaps
    z_j = Lambda(t_j) - Lambda(t_{j-1}) are iid Exp(1)."""
    col = "lam" if kind == "arr" else "mu"
    s = seg[seg.date.isin(days)].sort_values("start").reset_index(drop=True)
    e = ev[ev.date.isin(days)].sort_values("t")
    inc_rate = rate_lookup(rates, s.daytype, s.minute, col) * _avail(s.N, kind, C)
    cum = np.r_[0.0, np.cumsum(inc_rate * s.dur_h.values)]
    k = np.searchsorted(s.start.values, e.t.values, side="left") - 1
    k = np.clip(k, 0, len(s) - 1)
    dt_h = (e.t.values - s.start.values[k]).astype("timedelta64[ns]").astype(float) / 3.6e12
    Lam = cum[k] + inc_rate[k] * dt_h
    z = np.diff(Lam)
    ks = stats.kstest(z, "expon")
    lag1 = np.corrcoef(z[:-1], z[1:])[0, 1]
    return {"z": z, "n": len(z), "mean_z": z.mean(), "KS_stat": ks.statistic, "KS_p": ks.pvalue,
            "lag1_corr": lag1, "lag1_se": 1 / np.sqrt(len(z))}


def predictive_eval(occ, models: dict, C, train_days, test_days, holidays=None,
                    origins=range(6, 22), horizons=(1, 2, 3), rng=0):
    """From the observed N(t0), predict N(t0 + H).  models: name -> rates table
    (uses columns lam/mu, or lam_naive/mu_naive if the name ends in 'naive').
    Adds a 'climatology' baseline: the empirical distribution of N at that clock
    time on training days of the same type (ignores N(t0))."""
    rng = np.random.default_rng(rng)
    any_rates = next(iter(models.values()))
    bm = any_rates.attrs["bin_minutes"]
    mats = {}
    for name, r in models.items():
        cols = ("lam_naive", "mu_naive") if name.endswith("naive") else ("lam", "mu")
        for dt in ("weekday", "weekend"):
            if dt in r.index.get_level_values(0):
                mats[name, dt] = step_matrices(r, dt, C, *cols)

    def clim(dt, hour):
        days = train_days[daytype_of(train_days, holidays) == dt]
        n = np.clip(n_at(occ, days + pd.Timedelta(hours=hour)), 0, C)
        cnt = np.bincount(n, minlength=C + 1) + 0.5  # Jeffreys-style smoothing
        return cnt / cnt.sum()

    cache, rows = {}, []
    for d in test_days:
        dt = daytype_of([d], holidays)[0]
        for h0 in origins:
            for H in horizons:
                if h0 + H > 24:
                    continue
                n0 = min(int(n_at(occ, [d + pd.Timedelta(hours=h0)])[0]), C)
                n1 = min(int(n_at(occ, [d + pd.Timedelta(hours=h0 + H)])[0]), C)
                preds = {}
                for name in models:
                    key = (name, dt, h0, H)
                    if key not in cache:
                        cache[key] = transition_matrix(mats[name, dt], h0 * 60 // bm, (h0 + H) * 60 // bm)
                    preds[name] = cache[key][n0]
                ck = ("climatology", dt, h0 + H)
                if ck not in cache:
                    cache[ck] = clim(dt, h0 + H)
                preds["climatology"] = cache[ck]
                v = rng.random()
                for name, p in preds.items():
                    p = np.clip(p, 0, None) / p.sum()
                    cdf = np.cumsum(p)
                    lo, hi = np.searchsorted(cdf, 0.1), np.searchsorted(cdf, 0.9)
                    rows.append({"model": name, "date": d, "daytype": dt, "t0": h0, "H": H,
                                 "n0": n0, "n_obs": n1, "logscore": -np.log(max(p[n1], 1e-12)),
                                 "p_empty": p[0], "p_full": p[C], "mean_pred": (p * np.arange(C + 1)).sum(),
                                 "pit": (cdf[n1] - p[n1]) + v * p[n1],
                                 "covered80": lo <= n1 <= hi})
    out = pd.DataFrame(rows)
    out["abs_err"] = (out.mean_pred - out.n_obs).abs()
    return out


def calibration_table(pred, prob_col="p_empty", event="empty", C=None, bins=(0, .05, .1, .2, .3, .5, .7, 1.0001)):
    obs = pred.n_obs == 0 if event == "empty" else pred.n_obs == C
    g = pd.cut(pred[prob_col], bins, right=False)
    return pd.DataFrame({"predicted": pred[prob_col], "observed": obs}).groupby(g, observed=True).agg(
        n=("observed", "size"), mean_predicted=("predicted", "mean"), observed_freq=("observed", "mean"))


# ----------------------------------------------------------------------------
# 5. Synthetic data with known rates
# ----------------------------------------------------------------------------
def _bump(h, c, s):
    return np.exp(-((h - c) ** 2) / (2 * s ** 2))


def true_lambda(h, daytype):
    """Return-demand at an office-district station (per hour)."""
    base = 0.3 + 1.7 * ((h >= 6) & (h < 23.5))
    if daytype == "weekday":
        return base + 12 * _bump(h, 8.5, 0.8) + 3 * _bump(h, 14.0, 1.0) + 2 * _bump(h, 19.0, 1.5)
    return base + 4 * _bump(h, 12.5, 2.5)


def true_mu(h, daytype):
    """Pickup demand (per hour)."""
    base = 0.3 + 1.7 * ((h >= 6) & (h < 23.5))
    if daytype == "weekday":
        return base + 3 * _bump(h, 8.0, 1.0) + 3 * _bump(h, 14.5, 1.0) + 12 * _bump(h, 18.5, 0.9)
    return base + 4 * _bump(h, 15.0, 2.5)


def true_binned_rates(bin_minutes=60):
    rows = []
    for dt in ("weekday", "weekend"):
        for b in range(1440 // bin_minutes):
            h = np.linspace(b * bin_minutes / 60, (b + 1) * bin_minutes / 60, 61)
            rows.append({"daytype": dt, "bin": b, "lam_true": true_lambda(h, dt).mean(),
                         "mu_true": true_mu(h, dt).mean()})
    return pd.DataFrame(rows).set_index(["daytype", "bin"])


def simulate_trips(n_days=120, C=20, station="100", n_other=40, n_bikes_elsewhere=300,
                   bg_trips_per_day=900,
                   start="2024-01-01", day_sd=0.0, truck_level=10, truck_hour=3.0, seed=0):
    """Trips in EcoBici format for one target station with known demand.
    - pickup demand ~ Poisson(mu(t) * M_d), return demand ~ Poisson(lam(t) * M_d)
    - M_d = 1 if day_sd = 0, else lognormal daily multiplier (overdispersion)
    - pickups fail when empty, returns fail when full (censoring)
    - a truck resets the station to truck_level bikes every night (unobserved)
    Returns (trips DataFrame, true occupancy DataFrame(t, N))."""
    rng = np.random.default_rng(seed)
    others = [str(1000 + i) for i in range(n_other)]
    next_id = [10000]

    def new_bike():
        next_id[0] += 1
        return str(next_id[0])

    t_start = pd.Timestamp(start)
    at_target = [new_bike() for _ in range(truck_level)]
    elsewhere = {new_bike(): (rng.choice(others), t_start - pd.Timedelta(days=1))
                 for _ in range(n_bikes_elsewhere)}
    trips, truth = [], [(t_start, len(at_target))]

    def dur():
        return pd.Timedelta(minutes=float(np.clip(rng.lognormal(np.log(14), 0.5), 2, 90)))

    for day in range(n_days):
        d0 = t_start + pd.Timedelta(days=day)
        dt = "weekend" if d0.dayofweek >= 5 else "weekday"
        M = np.exp(rng.normal(0, day_sd) - day_sd ** 2 / 2) if day_sd > 0 else 1.0
        evs = []
        for kind, fn in (("ret", true_lambda), ("pick", true_mu)):
            rmax = 25.0 * M
            k = rng.poisson(rmax * 24)
            hs = np.sort(rng.uniform(0, 24, k))
            keep = rng.random(k) < fn(hs, dt) * M / rmax
            evs += [(h, kind) for h in hs[keep]]
        evs.append((truck_hour, "truck"))
        for h, kind in sorted(evs):
            t = d0 + pd.Timedelta(hours=float(h))
            if kind == "truck":
                while len(at_target) > truck_level:
                    b = at_target.pop(rng.integers(len(at_target)))
                    elsewhere[b] = (rng.choice(others), t)
                elig = [b for b, (_, ta) in elsewhere.items() if ta <= t]
                while len(at_target) < truck_level and elig:
                    b = elig.pop(rng.integers(len(elig)))
                    elsewhere.pop(b)
                    at_target.append(b)
            elif kind == "pick":
                if not at_target:
                    continue  # lost pickup (censored)
                b = at_target.pop(rng.integers(len(at_target)))
                dd = dur()
                dest = rng.choice(others)
                trips.append((b, station, t, dest, t + dd))
                elsewhere[b] = (dest, t + dd)
            else:  # return
                if len(at_target) >= C:
                    continue  # lost return (censored)
                dd = dur()
                for _ in range(20):
                    b = list(elsewhere)[rng.integers(len(elsewhere))]
                    if elsewhere[b][1] <= t - dd:
                        break
                else:
                    continue
                origin, _ = elsewhere.pop(b)
                trips.append((b, origin, t - dd, station, t))
                at_target.append(b)
            truth.append((t, len(at_target)))
        # background trips among other stations keep the rest of the fleet moving
        pool = list(elsewhere)
        for _ in range(rng.poisson(bg_trips_per_day * M)):
            b = pool[rng.integers(len(pool))]
            if b not in elsewhere:
                continue
            o, ta = elsewhere[b]
            t = max(ta, d0 + pd.Timedelta(hours=6)) + pd.Timedelta(hours=float(rng.exponential(2.0)))
            dd = dur()
            dest = rng.choice(others)
            trips.append((b, o, t, dest, t + dd))
            elsewhere[b] = (dest, t + dd)

    df = pd.DataFrame(trips, columns=["Bici", "Ciclo_Estacion_Retiro", "t_out", "Ciclo_EstacionArribo", "t_in"])
    df = df.sort_values("t_out")
    out = pd.DataFrame({
        "Genero_Usuario": rng.choice(["M", "F"], len(df)),
        "Edad_Usuario": rng.integers(18, 70, len(df)).astype(str),
        "Bici": df.Bici.values,
        "Ciclo_Estacion_Retiro": df.Ciclo_Estacion_Retiro.values,
        "Fecha_Retiro": df.t_out.dt.strftime("%d/%m/%Y").values,
        "Hora_Retiro": df.t_out.dt.strftime("%H:%M:%S").values,
        "Ciclo_EstacionArribo": df.Ciclo_EstacionArribo.values,
        "Fecha_Arribo": df.t_in.dt.strftime("%d/%m/%Y").values,
        "Hora_Arribo": df.t_in.dt.strftime("%H:%M:%S").values,
    })
    true_occ = pd.DataFrame(truth, columns=["t", "N"]).groupby("t", as_index=False).last()
    return out, true_occ

def enforce_capacity(stays, station, arrival_times, C, removable=("rebal_origin", "tail")):
    """Shorten truck-removal stays so that no arrival meets a full station.

    Two facts the data guarantees: N(t) <= C always, and an arrival at time t
    proves N(t-) < C. Walk through time keeping the count; whenever either
    fails, the truck must already have taken some bikes whose removal time we
    only guessed. End the oldest such stays (kind in `removable`) at t until
    the constraint holds. Stays seen directly in the trips ('observed') are
    never shortened.
    Returns (corrected stays, number of arrivals that still meet N >= C)."""
    import heapq
    st = stays[stays.station == station].copy()
    other = stays[stays.station != station]
    starts, ends = st.start.values, st.end.values.copy()
    can_cut = st.kind.isin(removable).values
    arr = np.sort(np.asarray(pd.to_datetime(arrival_times), dtype="datetime64[ns]"))
    # events: (time, order, kind, idx); at equal times: ends, then checks, then starts
    ev = [(t, 0, "end", i) for i, t in enumerate(ends)]
    ev += [(t, 1, "check", -1) for t in arr]
    ev += [(t, 2, "start", i) for i, t in enumerate(starts)]
    ev.sort(key=lambda x: (x[0], x[1]))
    present = np.zeros(len(st), bool)
    heap, N, unresolved = [], 0, 0          # heap: removable stays present, oldest first
    for t, _, kind, i in ev:
        if kind == "start":
            present[i] = True
            N += 1
            if can_cut[i]:
                heapq.heappush(heap, (starts[i], i))
            while N > C and heap:             # never more than C bikes docked
                _, j = heapq.heappop(heap)
                if present[j] and j != i:
                    ends[j] = t
                    present[j] = False
                    N -= 1
                elif j == i:                  # don't cut the bike that just arrived
                    heapq.heappush(heap, (starts[i], i))
                    break
        elif kind == "end":
            if present[i] and ends[i] == t:   # skip stays already ended early
                present[i] = False
                N -= 1
        else:  # arrival check
            while N >= C and heap:
                _, j = heapq.heappop(heap)
                if present[j]:
                    ends[j] = t
                    present[j] = False
                    N -= 1
            if N >= C:
                unresolved += 1
    st["end"] = ends
    st["idle_h"] = (st.end - st.start).dt.total_seconds() / 3600
    return pd.concat([other, st]).reset_index(drop=True), unresolved