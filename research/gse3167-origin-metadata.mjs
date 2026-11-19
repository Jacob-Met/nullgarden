// Official NCBI GEO GSE3167 provenance: streaming matrix-metadata audit.
// The parent series explicitly mentions repeated biopsies from one bladder.
// DO NOT infer subject independence from unique GSM accessions or title codes.
// Download as a wire-level source, hash compressed bytes, read only header
// metadata; the full matrix is drained to authenticate all source bytes.
import {createHash} from 'node:crypto';
import {createGunzip} from 'node:zlib';
import {Readable,Transform} from 'node:stream';
import {createInterface} from 'node:readline';

const source='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE3nnn/GSE3167/matrix/GSE3167_series_matrix.txt.gz';
const resp=await fetch(source,{signal:AbortSignal.timeout(180000)});
if(!resp.ok||!resp.body)throw new Error('GEO GSE3167 source fetch failed: HTTP '+resp.status+' from '+source);
const hash=createHash('sha256');
let compressedBytes=0, linesCount=0, tableRows=0, headerLineCount=0, inMatrix=false;
const tee=new Transform({transform(chunk,enc,cb){hash.update(chunk);compressedBytes+=chunk.length;cb(null,chunk)}});
const rl=createInterface({
  input:Readable.fromWeb(resp.body).pipe(tee).pipe(createGunzip()),
  crlfDelay:Infinity
});
const fields=new Map();
function unquote(s){
  let v=s.trim();if(v.length>=2&&v[0]==='"'&&v[v.length-1]==='"')v=v.slice(1,-1).replaceAll('""','"');
  return v.trim();
}
for await (const line of rl) {
  linesCount++;
  if(line.startsWith('!series_matrix_table_begin')){inMatrix=true;continue;}
  if(inMatrix){tableRows++;continue;}
  headerLineCount++;
  if(line.startsWith('!Sample_')){
    const cols=line.split('\t');if(cols.length<2)continue;
    const key=cols[0].trim();
    const values=cols.slice(1).map(unquote);
    if(!fields.has(key))fields.set(key,[]);
    fields.get(key).push(values);
  }
}
const field=(name)=>fields.get('!Sample_'+name)||[];
const access=field('geo_accession')[0],title=field('title')[0],sourceName=field('source_name_ch1')[0];
if(!access||!title||!sourceName)throw new Error('Official GEO matrix missing sample accession/title/source metadata');
if(access.length!==60||title.length!==60||sourceName.length!==60)
  throw new Error('Original GSE3167 series should have 60 arrays; metadata shape diverged');
const characteristics=field('characteristics_ch1');
const description=field('description');
for(const rows of [...characteristics,...description])if(rows.length!==60)
  throw new Error('Misaligned GEO sample metadata cannot infer correspondence');
const ids=new Set(access);
if(ids.size!==60)throw new Error('Duplicated official GSM accession');
const records=access.map((gsm,i)=>({
  gsm, title:title[i], sourceName:sourceName[i],
  characteristics:characteristics.map(r=>r[i]).filter(Boolean),
  description:description.map(r=>r[i]).filter(Boolean).slice(0,1)
}));
const explicitSubjectKeys=['patient id','patient_id','donor id','donor_id','subject id','subject_id'];
const idsFound=[];
for(const record of records){
  for(const field of record.characteristics){
    const idx=field.indexOf(':');
    if(idx>0&&explicitSubjectKeys.includes(field.slice(0,idx).trim().toLowerCase()))
      idsFound.push({gsm:record.gsm,key:field.slice(0,idx).trim(),publicSubjectCode:field.slice(idx+1).trim()});
  }
}
const output={
  schema:'nullgarden.geo-GSE3167-public-metadata.v1',source,
  compressedSha256:hash.digest('hex'),compressedBytes,linesCount,headerLineCount,
  dataMatrixRowsReadWithoutStoring:tableRows,
  numberOfGsmSamples:records.length,
  presentHeaderKeys:[...fields.keys()].sort(),
  explicitSubjectCodeCount:idsFound.length,explicitSubjectCodes:idsFound,
  parentStudyDescription:'GSE3167 public summary states some histologically normal and CIS biopsies came from the same urinary bladder',
  inference:'GSM accession identifies an assay, not an independent biological donor; no subject-level permutation validity may be inferred from accession uniqueness',
  records
};
console.log('GEO_ORIGIN_AUDIT='+JSON.stringify(output));
