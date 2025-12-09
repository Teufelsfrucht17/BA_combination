"""Create plots for DAX/SDAX portfolios, indices, and structure overview."""

from Plotting import (
    plot_portfolio_price_line,
    plot_index_price_line,
    plot_portfolio_structure_overview,
)


if __name__ == "__main__":
    # DAX portfolio of stocks (all stocks we consider, no index)
    plot_portfolio_price_line("dax", period_type="daily")

    # DAX portfolio of stocks excluding Rheinmetall (RHMG.DE)
    plot_portfolio_price_line("dax", period_type="daily", exclude_symbols=["RHMG.DE"])

    # SDAX portfolio of stocks (all stocks we consider, no index)
    plot_portfolio_price_line("sdax", period_type="daily")

    # DAX index only
    plot_index_price_line("dax", period_type="daily")

    # SDAX index only
    plot_index_price_line("sdax", period_type="daily")

    # Structure overview plots (prices, returns, dispersion, correlation)
    # Daily
    plot_portfolio_structure_overview("dax", period_type="daily")
    plot_portfolio_structure_overview("sdax", period_type="daily")

    # Intraday
    plot_portfolio_structure_overview("dax", period_type="intraday")
    plot_portfolio_structure_overview("sdax", period_type="intraday")
