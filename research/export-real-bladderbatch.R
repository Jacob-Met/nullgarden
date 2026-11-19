# Convert original measured Affymetrix probe expression values, not the
# synthetic expression generator that shares this study's batch design.
# Source: Bioconductor 3.23 experiment package bladderbatch 1.50.0.
# Requires Biobase + bladderbatch. No labels are used in probe selection.
suppressPackageStartupMessages(library(Biobase))
suppressPackageStartupMessages(library(bladderbatch))
data("bladderdata", package="bladderbatch")
if (!exists("bladderEset", inherits=FALSE)) stop("Missing official bladderEset")
if (as.character(packageVersion("bladderbatch")) != "1.50.0")
  stop("Unpinned bladderbatch release; must be 1.50.0")
expr <- Biobase::exprs(bladderEset)
pheno <- Biobase::pData(bladderEset)
stopifnot(is.matrix(expr), ncol(expr)==57, nrow(pheno)==57,
          identical(colnames(expr), rownames(pheno)),
          all(c("batch","cancer","sample","outcome") %in% names(pheno)),
          all(is.finite(expr)))
label_values <- sort(unique(as.character(pheno$cancer)))
cat("observed_original_cancer_values=",paste(label_values,collapse="|"),"\n",sep="")
cat("observed_original_cancer_counts=",paste(capture.output(print(table(pheno$cancer,useNA="ifany"))),collapse="; "), "\n",sep="")
cat("observed_original_outcome_counts=",paste(capture.output(print(table(pheno$outcome,useNA="ifany"))),collapse="; "), "\n",sep="")
if (!identical(label_values,c("Biopsy","Cancer","Normal")))
  stop("Pinned three-level bladderbatch phenotype changed; requalify the cancer/normal/biopsy mapping")
raw_pheno <- as.character(pheno$cancer)
# Primary contrast deliberately merges two DISTINCT reference phenotypes.
# The meaning is Cancer vs not-annotated-Cancer, not healthy-vs-disease.
binary <- as.integer(raw_pheno == "Cancer")
batch <- as.character(pheno$batch)
if (length(unique(batch))!=5) stop("Expected five processing batches")
if (length(unique(rownames(pheno)))!=57) stop("Nonunique original observation IDs")
# Probe selection independent of cancer labels and sampled processing dates.
# Freeze base R sampling and sort selected index for stable column order.
RNGkind(kind="Mersenne-Twister", normal.kind="Inversion", sample.kind="Rejection")
# The pilot 20261118 is retained (and SHA-checked downstream), followed by
# five additional fixed probe-panel seeds BEFORE their results are examined.
panel_seeds <- c(20261118L,20261119L,20261120L,20261121L,20261122L,20261123L)
output <- Sys.getenv("NULLGARDEN_REAL_CSV")
if (!nzchar(output)) stop("NULLGARDEN_REAL_CSV destination is required")
options(digits=17)
export_case <- function(name, keep, labels, path) {
  csv <- data.frame(sample_id=rownames(pheno)[keep],batch=batch[keep],
                    label=labels[keep],chosen[keep,,drop=FALSE],
                    check.names=FALSE,stringsAsFactors=FALSE)
  colnames(csv) <- c("sample_id","batch","label",paste0("probe",picks))
  write.table(csv,file=path,sep=",",row.names=FALSE,col.names=TRUE,
              quote=FALSE,na="")
  cat("contrast=",name," rows=",nrow(csv)," batches=",length(unique(csv$batch)),
      " reference_label0=",sum(csv$label==0),
      " cancer_label1=",sum(csv$label==1),"\n",sep="")
  for (b in sort(unique(csv$batch))) {
    counts <- table(factor(csv$label[csv$batch==b],levels=0:1))
    cat("contrast=",name," batch=",b," n_reference=",counts[1],
        " n_cancer=",counts[2],"\n",sep="")
  }
}
for (panel_seed in panel_seeds) {
  set.seed(panel_seed)
  picks <- sort(sample.int(nrow(expr), 200L, replace=FALSE))
  chosen <- t(expr[picks,,drop=FALSE])
  panel_path <- if (panel_seed==panel_seeds[1]) output
                else paste0(output,".panel-",panel_seed)
  cat("probe_panel_seed=",panel_seed," n_probes=",length(picks),"\n",sep="")
  export_case(paste0("seed-",panel_seed,"-pooled-reference"),
              rep(TRUE,57),binary,panel_path)
  export_case(paste0("seed-",panel_seed,"-normal-only"),
              raw_pheno %in% c("Cancer","Normal"),
              binary,paste0(panel_path,".cancer-v-normal.csv"))
  export_case(paste0("seed-",panel_seed,"-biopsy-only"),
              raw_pheno %in% c("Cancer","Biopsy"),
              binary,paste0(panel_path,".cancer-v-biopsy.csv"))
}
cat("source_package=bladderbatch_1.50.0 (Bioconductor 3.23)\n")
cat("expression_provenance=original RMA-normalized bladderEset assays\n")
cat("n_rows=",ncol(expr)," n_probes=",length(picks),
    " n_batches=",length(unique(batch)),"\n",sep="")
cat("cancer_positive_label=Cancer; reference_in_primary=Biopsy+Normal; no assertion of reference-tissue comparability\n")
cat("identity_limitation=sample IDs are not verified donor identities; no subject exchangeability is established\n")
cat("probe_selection=frozen six seeds 20261118..20261123; each unlabeled sample of 200 of ",nrow(expr)," probes\n",sep="")
