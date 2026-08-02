\begin{table}[t]

\centering
\caption{Quantitative comparison on long video generation. We evaluate Pyramid Forcing against autoregressive video diffusion generation baselines for 30- and 60-second videos using VBench-Long.}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccccc}
\hline
Model & Dynamic & Motion & Overall & Imaging & Aesthetic & Quality & Semantic & Total \\
& Degree $\uparrow$ & Smoothness $\uparrow$ & Consistency $\uparrow$ & Quality $\uparrow$ & Quality $\uparrow$ & Score $\uparrow$ & Score $\uparrow$ & Score $\uparrow$ \\
\hline
\multicolumn{9}{c}{\textit{30 seconds}} \\
CausVid & 45.70 & 98.25 & 22.39 & 66.11 & 59.63 & 86.66 & 50.93 & 79.51 \\
Rolling Forcing & 30.91 & 98.72 & 24.85 & 71.43 & 61.69 & 86.18 & 55.20 & 79.98 \\
LongLive & 41.70 & 98.79 & 24.54 & 68.79 & 61.36 & 86.97 & 54.31 & 80.44 \\
Self Forcing & 44.34 & 98.52 & 24.66 & 70.66 & 63.22 & 87.14 & 54.31 & 80.57 \\
\quad + Deep Forcing & 46.48 & 98.43 & 25.10 & 71.70 & 63.77 & 87.23 & 54.00 & 80.59 \\
\rowcolor{gray!15} \quad + Pyramid Forcing & 55.07 & 98.82 & 25.26 & 72.17 & 66.55 & 88.93 & 55.52 & 82.25 \\
Causal Forcing & 61.38 & 98.48 & 23.03 & 69.66 & 58.22 & 86.45 & 53.12 & 79.78 \\
\rowcolor{gray!15} \quad + Pyramid Forcing & 82.67 & 97.23 & 23.21 & 68.64 & 60.03 & 86.98 & 54.07 & 80.40 \\
\hline
\multicolumn{9}{c}{\textit{60 seconds}} \\
CausVid & 43.67 & 98.28 & 21.65 & 65.54 & 59.34 & 86.58 & 49.87 & 79.23 \\
Rolling Forcing & 32.08 & 98.73 & 24.20 & 70.55 & 61.06 & 86.19 & 54.20 & 79.80 \\
LongLive & 40.93 & 98.77 & 24.54 & 68.83 & 61.70 & 86.89 & 54.79 & 80.47 \\
Self Forcing & 43.75 & 97.85 & 22.33 & 64.28 & 55.62 & 84.84 & 50.01 & 77.87 \\
\quad + Deep Forcing & 43.10 & 98.35 & 24.48 & 68.43 & 60.48 & 86.37 & 54.91 & 80.08 \\
\rowcolor{gray!15} \quad + Pyramid Forcing & 53.68 & 98.53 & 24.70 & 70.03 & 62.00 & 87.58 & 55.71 & 81.21 \\
Causal Forcing & 57.03 & 98.49 & 22.51 & 68.63 & 56.87 & 85.84 & 52.36 & 79.14 \\
\rowcolor{gray!15} \quad + Pyramid Forcing & 86.39 & 97.03 & 22.41 & 68.38 & 58.91 & 86.64 & 53.05 & 79.92 \\
\hline
\end{tabular}
}
\label{tab:long_video_compare}
\end{table}