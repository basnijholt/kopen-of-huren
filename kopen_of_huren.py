from collections import defaultdict
from functools import partial
from itertools import product
from typing import Any, Dict, Literal, Union
from numbers import Number

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loky import get_reusable_executor
from tqdm.notebook import tqdm

from maandlasten import maandlasten
from mortgage import Mortgage

matplotlib.rc("font", size=15)


def load_sp500() -> pd.Series:
    # Daily data to need to resample it to quarterly like the huizenprijzen
    df_stock = pd.read_csv("sp500.csv")
    df_stock.Date = pd.to_datetime(df_stock.Date)
    df_stock.set_index("Date", inplace=True)
    # *Close price adjusted for splits
    # **Adjusted close price adjusted for both dividends and splits.
    stock_price = df_stock["Close*"].str.replace(",", "").astype(float)
    # Create data points for each day
    stock_price = stock_price.resample("D").interpolate()
    return stock_price


def plot_sp500() -> None:
    stock_price = load_sp500()
    stock_price.plot(
        xlabel="Datum",
        ylabel="S&P500 prijs ($)",
        title="S&P500 index vs. tijd, bron: Yahoo! Finance",
        figsize=(7, 7),
    )
    plt.show()


def get_groei() -> pd.DataFrame:
    stock_price = load_sp500()
    stock_price = stock_price[
        stock_price.index.day == 1
    ]  # Keep only first of the month
    first_year = stock_price.index.min().year
    start = f"{first_year+1}-02-01"

    stock_relative = {}
    for date, value in stock_price[stock_price.index >= start].items():
        date_prev = date.replace(date.year - 1)
        prev = stock_price[date_prev]
        stock_relative[date] = (value - prev) / prev * 100
    stock_relative = pd.Series(stock_relative)
    # Select at same dates as huis prijzen
    huis_relative = load_huizenprijzen()
    stock_relative = stock_relative[huis_relative.index]
    groei = pd.concat(
        [huis_relative, stock_relative], axis=1, keys=["huis", "aandelen"]
    )
    return groei


def plot_aandelen(groei: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    groei.aandelen.plot(
        ax=ax,
        xlabel="Datum",
        ylabel="S&P500 prijs stijging/daling per jaar (%)",
        title="S&P500 index vs. tijd, bron: Yahoo! Finance",
        color="k",
    )
    fill_area(groei.aandelen, ax)
    plt.show()


def load_huizenprijzen():
    # Load the data
    df_huis = pd.read_csv("huizenprijzen.csv", delimiter=";")
    df_huis.Perioden = pd.to_datetime(
        df_huis.Perioden.str.replace("e kwartaal", "").str.replace(" ", "-Q")
    )
    df_huis.set_index("Perioden", inplace=True)

    # Interpolate to daily and than select first days of the month
    huis_relative = (
        df_huis["Prijsindex verkoopprijzen/Ontwikkeling  t.o.v. een jaar eerder (%)"]
        .resample("D")
        .interpolate()
    )
    huis_relative = huis_relative[huis_relative.index.day == 1]
    huis_relative = huis_relative[~huis_relative.isna()]  # Drop NaNs
    return huis_relative


def plot_huizenprijzen(groei: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    groei.huis.plot(
        ax=ax,
        legend=False,
        xlabel="Datum",
        ylabel="Huizenprijs stijging/daling per jaar (%)",
        title="Huizenprijs verschil vs. tijd, bron: CBS",
        figsize=(8, 8),
        color="k",
    )
    fill_area(groei.huis, ax)
    plt.show()


def plot_aandelen_en_huis(groei: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    groei.aandelen[groei.huis.index].plot(ax=ax, label="Aandelen", legend=True)
    groei.huis.plot(ax=ax, label="Huizenprijs", legend=True)
    ax.set_title("Huizenprijs en aandelenprijs stijging/daling per jaar in %")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Prijs stijging/daling per jaar (%)")
    fill_area(groei.aandelen, ax, alpha=0.3)
    fill_area(groei.huis, ax, alpha=0.3)
    plt.show()


def vergelijkings_tabel(groei: pd.DataFrame):
    example_periods = [
        dict(van="2014-Q2", tot="2020-Q4", notities="de recente 'goede' jaren"),
        dict(
            van="2009-Q2", tot="2014-Q1", notities="slechtste jaren na de 2008 crisis"
        ),
        dict(van="2009-Q2", tot="2020-Q4", notities="van 2008 crisis tot en met nu"),
        dict(
            van="1996-Q1", tot="2020-Q4", notities="alle data sinds 1996 tot en met nu"
        ),
    ]
    for dct in example_periods:
        mean = lambda x: x[(x.index >= dct["van"]) & (x.index <= dct["tot"])].mean()
        dct["huis"] = f"{mean(groei.huis):.2f}%"
        dct["aandelen"] = f"{mean(groei.aandelen):.2f}%"
        winner = "huis" if mean(groei.huis) > mean(groei.aandelen) else "aandelen"
        dct[winner] += " 🏆"
        dct["verschil (🏠 - 📈)"] = f"{mean(groei.huis) - mean(groei.aandelen):.2f}%"
        dt = (pd.to_datetime(dct["tot"]) - pd.to_datetime(dct["van"])).total_seconds()
        dct["lengte periode"] = f"{round(dt / 86400 / 365)} jaar"

    table = pd.DataFrame(example_periods)[
        [
            "van",
            "tot",
            "lengte periode",
            "huis",
            "aandelen",
            "verschil (🏠 - 📈)",
            "notities",
        ]
    ]
    return table


def fill_area(x: pd.Series, ax, alpha: float = 1.0) -> None:
    ax.fill_between(
        x.index,
        x.values,
        where=x.values > 0,
        color="green",
        alpha=alpha,
        zorder=-1,
    )
    ax.fill_between(
        x.index,
        x.values,
        where=x.values < 0,
        color="red",
        alpha=alpha,
        zorder=-1,
    )
    ax.hlines(0, x.index.min(), x.index.max(), ls="--", color="k")


def maandelijke_groei(
    date: pd.Timestamp, groei: pd.DataFrame, which: Literal["huis", "aandelen"] = "huis"
) -> float:
    pct = groei[which][groei.index == date].iloc[0] / 100
    return (1 + pct) ** (1 / 12)


def bepaal_woz(huidige_prijs: float, date: pd.Timestamp, groei: pd.DataFrame):
    """WOZ waarde is bepaald aan de hand van de prijs van vorig jaar."""
    vorig_jaar = date.year - 1
    dates = groei.index[groei.index.year == vorig_jaar]
    prijs = huidige_prijs
    for _date in dates[::-1]:
        # We rekenen terug naar de prijs van vorig jaar
        prijs /= maandelijke_groei(_date, groei, "huis")
    return prijs


def aantal_jaar(dates: pd.DatetimeIndex):
    dt = dates.max() - dates.min()
    return dt.total_seconds() / 86400 / 365.25


def maandelijks_onderhoud(huis_waarde: float, onderhoud_pct: float = 2):
    return huis_waarde * onderhoud_pct / 100 / 12


def vermogensbelasting(
    vermogen: float, schulden: float = 0, met_fiscaal_partner: bool = True
):
    """Vermogensbelasting vanaf 2021.
    https://www.rijksoverheid.nl/onderwerpen/belastingplan/belastingwijzigingen-voor-ons-allemaal/box-3
    """
    heffingvrij = 100_000 if met_fiscaal_partner else 50_000
    vermogen -= heffingvrij
    vermogen -= schulden
    if vermogen < 0:
        return 0
    # De rest is in box 3
    schijf_1 = 100_000 - 50_000
    belastbaar_1 = min(vermogen, schijf_1)
    vermogen -= belastbaar_1
    inkomen_1 = belastbaar_1 * 1.90 / 100

    schijf_2 = 1_000_000 - 100_000
    belastbaar_2 = min(vermogen, schijf_2)
    vermogen -= belastbaar_2
    inkomen_2 = belastbaar_2 * 4.50 / 100

    schijf_3 = float("inf")
    belastbaar_3 = min(vermogen, schijf_3)
    vermogen -= belastbaar_3
    inkomen_3 = belastbaar_3 * 5.69 / 100

    inkomen = inkomen_1 + inkomen_2 + inkomen_3
    return inkomen * 31 / 100


def koop_huis_of_beleg(
    aankoop_datum: Union[str, pd.Timestamp],
    jaar_tot_verkoop: Number,
    geleend: Number,
    groei: pd.DataFrame,
    huur: Number = 1000,
    hypotheekrente: Number = 2.04,
    hyptotheek_looptijd: int = 30 * 12,
    jaarinkomen: Number = 90_000,
    schulden: Number = 20_000,
    onderhoud_pct: Number = 1,
    met_fiscaal_partner: bool = True,
    verbose: bool = True,
):
    dates = groei.index[groei.index >= aankoop_datum][
        : round(jaar_tot_verkoop * 12) + 1
    ]

    if len(dates) < jaar_tot_verkoop * 12:
        raise ValueError(
            f"Een duur van {jaar_tot_verkoop} jaar is niet mogelijk als "
            f"we starten op {aankoop_datum}. "
            f"Een duur van {aantal_jaar(dates):.2f} is mogelijk."
        )

    persoon = maandlasten.Persoon(jaarinkomen)
    onderhoud = partial(maandelijks_onderhoud, onderhoud_pct=onderhoud_pct)
    hypotheek = Mortgage(hypotheekrente / 100, hyptotheek_looptijd, geleend)
    betaalschema = hypotheek.monthly_payment_schedule()
    rente_betaald: Dict[int, float] = defaultdict(float)
    start_year = dates[0].year

    betaald = 0
    afgelost = 0
    belegging = 0
    huis_waarde = geleend
    for date in dates:
        huis_waarde *= maandelijke_groei(date, groei, "huis")
        belegging *= maandelijke_groei(date, groei, "aandelen")
        betaald += onderhoud(huis_waarde)
        afbetaling, rente = next(betaalschema)
        hypotheek_kosten = float(afbetaling) + float(rente)
        rente_betaald[date.year] += float(rente)
        betaald += hypotheek_kosten
        belegging += hypotheek_kosten - huur
        afgelost += float(afbetaling)
        if date.month == 1 and date.year > start_year:
            # Betaal vermogensbelasting over vorig jaar
            belegging -= vermogensbelasting(belegging, schulden, met_fiscaal_partner)
            # Krijg hypotheekrenteaftrek terug van vorig jaar!
            woz_waarde = bepaal_woz(huis_waarde, date, groei)
            hypotheek_aftrek = maandlasten.hypotheek_aftrek(
                rente_betaald[date.year - 1], woz_waarde
            )
            persoon_met_aftrek = maandlasten.Persoon(persoon.bruto_jaarloon)
            persoon_met_aftrek.aftrek = hypotheek_aftrek
            teruggave = persoon_met_aftrek.netto_loon - persoon.netto_loon
            betaald -= teruggave

    af_te_lossen = geleend - afgelost
    overdrachts_belasting = huis_waarde * 0.02
    huis_winst = huis_waarde - af_te_lossen - betaald - overdrachts_belasting

    if verbose:
        winst_of_verlies = "winst" if huis_winst > 0 else "verlies"
        print(
            f"We hebben op {aankoop_datum} een huis van €{geleend/1000:.0f}k gekocht. "
            f"Op {date.date()} (na {aantal_jaar(dates):.1f} jaar) hebben we €{betaald/1000:.0f}k betaald, "
            f"€{afgelost/1000:.0f}k afgelost, een huiswaarde van €{huis_waarde/1000:.0f}k, "
            f"en na een verkoop €{abs(huis_winst)/1000:.0f}k {winst_of_verlies}. "
            f"Hadden we een huis gehuurd voor €{huur} per maand en belegd, dan hadden we €{belegging/1000:.0f}k. "
            f"Dat is dus €{(belegging - huis_winst)/1000:.0f}k verschil."
        )

    return dict(
        aankoop_datum=aankoop_datum,
        verkoop_datum=dates[-1],
        aantal_jaar=aantal_jaar(dates),
        betaald=betaald,
        afgelost=afgelost,
        af_te_lossen=af_te_lossen,
        huis_waarde=huis_waarde,
        huis_winst=huis_winst,
        belegging=belegging,
    )


def run_monte_carlo(groei: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    start_jaar = groei.index.year.min() + 1
    eind_jaar = groei.index.year.max()
    n_jaar = eind_jaar - start_jaar + 1

    results = {}
    iterator = list(
        product(groei.index[groei.index.year >= start_jaar], range(1, n_jaar))
    )

    def try_run_simulation(datum_jaar, parameters):
        aankoop_datum, jaar_tot_verkoop = datum_jaar
        try:
            return koop_huis_of_beleg(
                aankoop_datum, jaar_tot_verkoop, groei=groei, **parameters
            )
        except ValueError:
            # 'jaar' is niet mogelijk want we kunnen niet in de toekomst kijken
            return

    with get_reusable_executor() as executor:
        results = list(
            tqdm(
                executor.map(
                    partial(try_run_simulation, parameters=parameters), iterator
                ),
                "Monte Carlo simulatie",
                total=len(iterator),
            )
        )

    df = pd.DataFrame([r for r in results if r is not None])
    df.aankoop_datum = pd.to_datetime(df.aankoop_datum)
    df["verschil"] = (df.huis_winst - df.belegging) / 1000
    df.aantal_jaar = df.aantal_jaar.round()
    return df


def plot_result_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    df.plot.scatter(
        ax=ax,
        x="aankoop_datum",
        y="aantal_jaar",
        c="verschil",
        s=100,
        alpha=1,
        norm=matplotlib.colors.TwoSlopeNorm(0),
        cmap="seismic",
        title="Kopen of huren?",
        xlabel="Aankoop datum",
        ylabel="verkopen na (jaar)",
        figsize=(8, 8),
    )
    ax, cax = plt.gcf().get_axes()
    cax.set_ylabel("verschil (x€1000)")
    ax.text(
        0.95,
        0.95,
        "rood is huis is beter\nblauw is belegging is beter",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        fontsize=14,
    )
    plt.show()


def plot_result_contour(df: pd.DataFrame) -> None:
    ds = df.set_index(["aantal_jaar", "aankoop_datum"]).to_xarray()
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8), sharex=True, sharey=True)
    levels = 15
    ds.verschil.plot.contourf(
        ax=axs[0, 0],
        norm=matplotlib.colors.TwoSlopeNorm(
            0, vmin=ds.verschil.min(), vmax=ds.verschil.max()
        ),
        add_colorbar=True,
        levels=levels,
        cbar_kwargs={"label": "Verschil (x€1000)"},
    )
    (ds.belegging / 1000).plot.contourf(
        ax=axs[0, 1],
        add_colorbar=True,
        levels=levels,
        cbar_kwargs={"label": "Waarde belegging (x€1000)"},
    )
    (ds.huis_winst / 1000).plot.contourf(
        ax=axs[1, 0],
        add_colorbar=True,
        levels=levels,
        norm=matplotlib.colors.TwoSlopeNorm(
            0, vmin=ds.huis_winst.min() / 1000, vmax=ds.huis_winst.max() / 1000
        ),
        cbar_kwargs={"label": "Winst vrkp huis (x€1000)"},
    )
    (ds.huis_waarde / 1000).plot.contourf(
        ax=axs[1, 1],
        add_colorbar=True,
        cbar_kwargs={"label": "Huis waarde (x€1000)"},
        cmap="magma",
        levels=levels,
    )

    axs[0, 0].text(
        0.95,
        0.95,
        "rood is huis is beter\nblauw is belegging is beter",
        horizontalalignment="right",
        verticalalignment="top",
        transform=axs[0, 0].transAxes,
        fontsize=12,
    )

    axs[1, 0].set_xlabel("Aankoop datum")
    axs[1, 1].set_xlabel("Aankoop datum")
    axs[0, 0].set_ylabel("Verkoop na (jaar)")
    axs[1, 0].set_ylabel("Verkoop na (jaar)")
    axs[0, 0].set_xlabel("")
    axs[0, 1].set_xlabel("")
    axs[0, 1].set_ylabel("")
    axs[1, 1].set_ylabel("")
    plt.show()


def plot_result_lines(df: pd.DataFrame) -> None:
    jaren = df.aantal_jaar.unique()[1::2]
    cmap = matplotlib.cm.get_cmap("tab20", len(jaren))
    color_map = dict(zip(sorted(jaren), cmap.colors))

    fig, ax = plt.subplots(figsize=(8, 8))
    for jaar in jaren:
        df[df.aantal_jaar == jaar].plot(
            x="aankoop_datum", y="verschil", ax=ax, color=color_map[jaar], legend=False
        )
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(cmap=cmap),
        ax=ax,
    )
    cbar.set_ticks(np.linspace(0, 1, len(jaren)))
    cbar.set_ticklabels([int(j) for j in color_map.keys()])
    cbar.set_label("Verkoop na (jaar)")
    ax.hlines(
        0, df.aankoop_datum.min(), df.aankoop_datum.max(), ls="--", color="k", zorder=-1
    )
    ax.set_xlabel("Aankoop datum")
    ax.set_ylabel("Winst kopen huis t.o.v. beleggen")
    ax.set_title("Winst kopen huis t.o.v. beleggen")
    plt.show()
