# Observable 57-row Bioconductor source-to-GEO accession bridge.
# Preserves assay identifiers ONLY; no gene expression / patient attributes.
# The label 'sample' is a source annotation, NOT a verified biological donor.
suppressPackageStartupMessages(library(Biobase))
suppressPackageStartupMessages(library(bladderbatch))
data("bladderdata", package="bladderbatch")
if(!exists("bladderEset",inherits=FALSE))stop("Missing original Bioconductor bladderEset")
if(as.character(packageVersion("bladderbatch"))!="1.50.0")
  stop("Unpinned bladderbatch release")
p <- Biobase::pData(bladderEset)
stopifnot(nrow(p)==57, all(c("batch","cancer","sample") %in% names(p)),
          identical(rownames(p), colnames(Biobase::exprs(bladderEset))))
out <- Sys.getenv("NULLGARDEN_BIOCONDUCTOR_TSV")
if(!nzchar(out))stop("NULLGARDEN_BIOCONDUCTOR_TSV is required")
if(anyNA(p$sample)||anyNA(rownames(p)))stop("Missing original annotation")
rows <- data.frame(assay_id=rownames(p),
   sample_annotation=as.character(p$sample),
   batch=as.character(p$batch),
   phenotype=as.character(p$cancer),stringsAsFactors=FALSE)
stopifnot(all(!grepl("[\t\r\n]",unlist(rows))),
          length(unique(rows$assay_id))==57)
write.table(rows,file=out,sep="\t",quote=FALSE,row.names=FALSE,col.names=TRUE)
cat("BLA_DONOR_CROSSWALK_SOURCE=bladderbatch_1.50.0, n=",nrow(rows),
    ", distinct_assay_ids=",length(unique(rows$assay_id)),
    ", distinct_sample_annotations=",length(unique(rows$sample_annotation)),
    "\n",sep="")
for(i in seq_len(nrow(rows)))
 cat("BLA_ASSAY_ROW\t",rows$assay_id[i],"\t",rows$sample_annotation[i],
     "\t",rows$batch[i],"\t",rows$phenotype[i],"\n",sep="")
