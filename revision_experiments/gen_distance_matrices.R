#!/usr/bin/env Rscript
# Regenerate the DD-NS diffusion-distance matrices used by neg_sample="commute_distance".
#
# Faithful to LightGCN-PyTorch/code/sources/hitting_time_calculate.R:
#   diffudist::get_distance_matrix(g, tau, type = "Normalized Laplacian")
# on the undirected bipartite user-item graph (nodes labelled u_* / i_*).
#
# NOTE: diffudist 1.0.1's default type IS "Normalized Laplacian" (verified), so we use the
# default and avoid passing the string explicitly (the package's own type-matching has an
# encoding quirk when strex is absent; using the default sidesteps any ambiguity).
#
# Run inside the conda env r_diffudist (R 4.4 + igraph + diffudist 1.0.1 + strex):
#   conda run -n r_diffudist Rscript gen_distance_matrices.R <edge_list_csv> <out_dir> <tau1> [tau2 ...]
# Output: <out_dir>/distance_tau<tau>.csv  (first col "" -> pandas "Unnamed: 0"; cols = node ids)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("usage: Rscript gen_distance_matrices.R <edge_list_csv> <out_dir> <tau1> [tau2 ...]")
edge_csv <- args[1]
out_dir  <- args[2]
taus     <- as.numeric(args[-(1:2)])

suppressMessages({ library(igraph); library(diffudist) })
cat(sprintf("[env] diffudist %s\n", as.character(packageVersion("diffudist"))))

cat(sprintf("[read] %s\n", edge_csv)); flush.console()
data <- read.csv(edge_csv)
g <- graph_from_data_frame(data, directed = FALSE)
cat(sprintf("[graph] %d nodes, %d edges\n", vcount(g), ecount(g))); flush.console()

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
for (tau in taus) {
  cat(sprintf("[diffdist] tau=%g computing ...\n", tau)); flush.console()
  t0 <- Sys.time()
  dist <- get_distance_matrix(g, tau = tau, verbose = FALSE)   # default type = Normalized Laplacian
  outf <- file.path(out_dir, sprintf("distance_tau%s.csv", format(tau, trim = TRUE)))
  write.csv(dist, outf)
  dt <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  cat(sprintf("[diffdist] tau=%g wrote %s (%.1f MB) in %ss\n",
              tau, outf, file.info(outf)$size / 1e6, dt)); flush.console()
}
cat("[done] all matrices generated\n")