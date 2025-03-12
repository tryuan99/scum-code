"""The stdin live plotter is used to plot live data from standard input."""

from collections.abc import Callable

from utils.plotter.live_plotter import ContinuousLivePlotter


class StdinLivePlotter(ContinuousLivePlotter):
    """Standard input live plotter to plot data from standard input."""

    def __init__(
        self,
        max_duration: float,
        parse_data: Callable[[str], float | list[float]],
        title: str,
        xlabel: str,
        ylabel: str,
        ymin: float,
        ymax: float,
        num_traces: int = 1,
        trace_labels: list[str] = None,
        secindices: list[int] = None,
        secylabel: str = None,
        secymin: float = None,
        secymax: float = None,
    ) -> None:
        super().__init__(
            max_duration,
            title,
            xlabel,
            ylabel,
            ymin,
            ymax,
            num_traces=num_traces,
            trace_labels=trace_labels,
            secindices=secindices,
            secylabel=secylabel,
            secymin=secymin,
            secymax=secymax,
        )
        self.parse_data = parse_data

    def next(self) -> float | list[float]:
        """Returns the next y-value to plot.

        This function blocks until the next value is available.
        """
        read_data = input()
        while len(read_data) == 0:
            read_data = input()
            read_data = read_data.strip()
        return self.parse_data(read_data)
