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
if (length(label_values)!=2 || anyNA(label_values) || any(!nzchar(label_values)))
  stop("Expected two nonempty cancer factor groups")
binary <- as.integer(match(as.character(pheno$cancer), label_values)-1L)
batch <- as.character(pheno$batch)
if (length(unique(batch))!=5) stop("Expected five processing batches")
if (length(unique(rownames(pheno)))!=57) stop("Nonunique original observation IDs")
# Probe selection independent of cancer labels and sampled processing dates.
# Freeze base R sampling and sort selected index for stable column order.
RNGkind(kind="Mersenne-Twister", normal.kind="Inversion", sample.kind="Rejection")
set.seed(20261118)
picks <- sort(sample.int(nrow(expr), 200L, replace=FALSE))
chosen <- t(expr[picks,,drop=FALSE])
csv <- data.frame(sample_id=rownames(pheno),batch=batch,label=binary,
                  chosen,check.names=FALSE,stringsAsFactors=FALSE)
colnames(csv) <- c("sample_id","batch","label",paste0("probe",picks))
output <- Sys.getenv("NULLGARDEN_REAL_CSV")
if (!nzchar(output)) stop("NULLGARDEN_REAL_CSV destination is required")
write.table(csv,file=output,sep=",",row.names=FALSE,col.names=TRUE,
            quote=FALSE,na="")
cat("source_package=bladderbatch_1.50.0 (Bioconductor 3.23)\n")
cat("expression_provenance=original RMA-normalized bladderEset assays\n")
cat("n_rows=",nrow(csv)," n_probes=",length(picks),
    " n_batches=",length(unique(batch)),"\n",sep="")
cat("cancer_label_0=",label_values[1]," cancer_label_1=",label_values[2],"\n",sep="")
for (b in sort(unique(batch))) {
  counts <- table(factor(binary[batch==b],levels=0:1))
  cat("batch=",b," n_label0=",counts[1]," n_label1=",counts[2],"\n",sep="")
}
cat("identity_limitation=sample IDs are not verified donor identities; no subject exchangeability is established\n")
cat("probe_selection=frozen seed 20261118; unsupervised 200 of ",nrow(expr)," probes\n",sep="")
