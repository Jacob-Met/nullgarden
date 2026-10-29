"""Create a fresh synthetic dataset for schedule mechanics, with no downloads."""
import argparse, hashlib, json, pathlib
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    root=pathlib.Path(a.out)
    root.mkdir(parents=False,exist_ok=False)
    X=np.arange(18,dtype=np.float64).reshape(6,3)/7
    matrix=root/"dataset.npz"
    np.savez_compressed(matrix,X=X)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    meta={"schema":"nullgarden.dataset.v1","id":"schedule-mechanics-six",
          "name":"Six-sample schedule mechanics","kind":"authored",
          "y":[0,1,1,0,1,0],"groups":["A","A","A","B","B","C"],
          "sample_ids":["s"+str(i) for i in range(6)],
          "feature_names":["f0","f1","f2"],
          "provenance":{"kind":"authored","description":"Mechanical synthetic fixture; no scientific or biological claim."},
          "storage":{"npz_sha256":sha(matrix.read_bytes()),
                     "matrix_sha256":sha(np.ascontiguousarray(X,dtype="<f8").tobytes(order="C"))}}
    metadata=root/"dataset.json"
    with metadata.open("x",encoding="utf-8") as f:
        json.dump(meta,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    receipt={"dataset":str(root),"files":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in (matrix,metadata)},
             "expected":{"label_assignment_support":6,"frozen_labels":1,"max_hamming":4},
             "scientific_assumptions":"unverified"}
    with (root/"creation-receipt.json").open("x",encoding="utf-8") as f: json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,sort_keys=True))
if __name__=="__main__": main()
