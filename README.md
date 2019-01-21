# RAL-Net

1.This code is the ACCV2018 paper [Robust Angular Local Descriptor Learning](http://igm.univ-mlv.fr/~cwang/papers/ACCV2018_DescriptorLearning.pdf)

2.Enviromental built: pytorch 0.4

3.To replicate the result of this paper:

python RAL_Net.py --data-root=/mnt/Brain/wug/Brown_data/ --epochs=10 --batch-size=512 --n-pairs=5000000 --loss-type=RAL_loss --lr=10 --augmentation=True

Result:
\begin{table}[h]
\large
\centering
\renewcommand\arraystretch{1.4}
 \caption{Descriptor performance on Brown dataset for patch verification. False positive rate at 95\% true positive rate is displayed. Results of the best are in bold and "+" suffix represent training implemented by data augmentation of random flipping and $90^\circ$ rotating. }
 \resizebox{1\textwidth}{!}{
 \begin{tabular}{ccccc}
  \toprule
 Training{ } { } & \underline{Notredame{ } { } Yosemite{ } { }} & \underline{Liberty { } { } Yosemite{ } { }} & \underline{Liberty { } { } Notredame{ } { }} & { } { }Mean  \\
 Test & Liberty & Notredame & Yosemite & FPR \\
 \midrule
   SIFT\cite{Lowe2004} & 29.84 & 22.53 & 27.29 & 26.55  \\
   MatchNet \cite{7298948} & 7.04 { } { } 11.47 & 3.82 { } { } 5.65 & 11.6 { } { } 8.7 & 8.05  \\
   L2Net\cite{8100132} & 3.64 { } { } 5.29 & 1.15 { } { } 1.62 & 4.43 { } { } 3.30 & 3.23  \\
 L2Net${+}$\cite{8100132} & 2.36 { } { } 4.70 & 0.72 { } { } 1.29 & 2.57 { } { } 1.71 & 2.22\\
 CS L2Net\cite{8100132} & 2.55 { } { } 4.24 & 0.87 { } { } 1.39 & 3.81 { } { } 2.84 & 2.61\\
 CS L2Net${+}$\cite{8100132} & 1.71 { } { } 3.87 & 0.56 { } { } 1.09 & 2.07 { } { } 1.3 & 1.76\\
  HardNetNIPS \cite{hard} { } { } & 3.06 { } { } 4.27 & 0.96 { } { } 1.4 & 3.04 { } { } 2.53 & 2.54\\
 HardNet+NIPS \cite{hard} { } { } & 2.28 { } { } 3.25 & 0.57 { } { } 0.96 & 2.13 { } { } 2.22 & 1.9 \\
 \midrule
 \multicolumn{5}{c}{Traing strategy 1: 200K training pairs for each subset, batch size 128} \\
 \midrule
 HardNet$_{128}${ } { } &2.07  { } { } 3.70  &0.77  { } { }1.22  & 3.79 { } { } 3.33 &2.48  \\
 HardNet$_{128}$+{ } { } & 2.46 { } { } 3.55  & 0.73 { } { } 1.67 & 3.54 { } { } 3.40 & 2.56 \\
RAL-Net$_{128}$(ours){ } { } & \textbf {1.46} { } { } \textbf{2.63} &\textbf {0.51} { } { } \textbf{0.91} & \textbf{1.95} { } { }\textbf {1.40} &\textbf {1.48} \\
 RAL-Net$_{128}$+(ours){ } { } &  {1.81} { } { }{ 3.80} & {0.55} { } { } {1.01} & {1.96} { } { } {2.18} & {1.89} \\ 
 \midrule
 \multicolumn{5}{c}{Traing strategy 2: 5000k training pairs for each subset, batch size 512} \\
 \midrule
 HardNet$_{512}$  { } { } & 1.54 { } { } 2.56 & 0.63 { } { } 0.92 & 2.65 { } { } 2.05 & 1.73\\
 HardNet$_{512}$+  { } { } & 2.53 { } { } 2.69 & 0.54 { } { } 0.83 & 2.49 { } { } 1.70 & 1.80 \\
  HardNet$_{1024}$  { } { } & 1.47 { } { } 2.67 & 0.62 { } { } 0.88 & 2.14 { } { } 1.65 & 1.57\\
 HardNet$_{1024}$+  { } { } & 1.49 { } { } 2.51 & 0.53 { } { } 0.78 & 1.96 { } { } 1.84 & 1.51 \\
 RAL-Net$_{512}$(ours) { } { } & {1.44} { } { } {2.60} & {0.48} { } { } {0.77} & {1.77} { } { } {1.43} & {1.42}\\
 RAL-Net$_{512}$+(ours) { } { } &\textbf {1.30} { } { }\textbf {2.39} &\textbf {0.37} { } { }\textbf {0.67} &\textbf {1.52} { } { }\textbf {1.31} &\textbf {1.26}\\
  \bottomrule
 \end{tabular}
 }
 \label{Brown result}
\vspace{-0.2cm}
\end{table}

