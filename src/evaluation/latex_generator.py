from typing import List, Tuple

LINEPLOT_TEMPLATE = """
\\begin{figure}[!h]
    \\centering
    \\begin{subfigure}{.475\\linewidth}
    \\centering
    \\begin{tikzpicture}[yscale=0.9]
    \\begin{axis}[
        % title={{#TITLE#}},
        xlabel={#X_LABEL#},
        ylabel={AUROC},
        xmin=0, xmax={#X_MAX#},
        ymin=0, ymax=1,
        xtick={#X_TICKS#},
        ytick={0.0, 0.2, 0.4, 0.6, 0.8, 1.0},
        legend entries={#LEGEND_ENTRIES#},
        legend pos = outer north east,
        ymajorgrids=true,
        grid style=dashed,
        width=\\linewidth
    ]

#CONTENT#

    \\end{axis}
    \\end{tikzpicture}
    \\end{subfigure}

    \\caption{#CAPTION#}
    \\label{fig:#LATEX_LABEL#}
\\end{figure}
"""


NAMES_TRANSLATION = {
    'Recurrent EBM': '\\ebm',
    'LSTM-AD': '\\lstmad',
    'LSTMED': '\\lstmed',
    'Donut': '\\donut',
    'DAGMM_NNAutoEncoder_withWindow': '\\dagmm',
    'DAGMM_LSTMAutoEncoder_withWindow': '\\lstmdagmm',
    'AutoEncoder': '\\nnae',
}


class LatexGenerator:

    @staticmethod
    def generate_lineplot(title, x_ticks, detector_names, lines,
                          x_label='Dataset', caption='Some caption', latex_label='sample_label'):
        x_max = f'{float(x_ticks[-1]) * 1.1:.2f}'
        legend_entries = ', '.join([NAMES_TRANSLATION.get(x, x) for x in detector_names if x in NAMES_TRANSLATION])
        return LINEPLOT_TEMPLATE  \
            .replace('#TITLE#', title) \
            .replace('#X_LABEL#', x_label) \
            .replace('#X_TICKS#', ', '.join(x_ticks)) \
            .replace('#X_MAX#', x_max) \
            .replace('#LEGEND_ENTRIES#', legend_entries) \
            .replace('#CAPTION#', caption) \
            .replace('#LATEX_LABEL#', latex_label) \
            .replace('#CONTENT#', LatexGenerator._get_lines_coordinates(lines, detector_names))

    @staticmethod
    def _get_lines_coordinates(lines: List[List[Tuple[str, float]]], detector_names: List[str]):
        output = ''
        for line, det in zip(lines, detector_names):
            # e.g. skip DAGMM_NNAutoEncoder_withoutWindow
            if det not in NAMES_TRANSLATION:
                continue
            output += '\t\\addplot coordinates'
            output += f'{{{" ".join([f"({pollution}, {value:.2f})" for pollution, value in line])}}};'
            output += '\n'
        return output
