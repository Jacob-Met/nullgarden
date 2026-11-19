// Exact arithmetic over the *ingested IEEE-754 binary64 values*.
// The CSV's original decimal measurement has already been rounded by the
// numeric parser. No phantom decimal precision is claimed here.
const ZERO=Object.freeze({n:0n,d:1n});
const ONE=Object.freeze({n:1n,d:1n});
const negative=v=>v<0n?-v:v;
function gcd(a,b){
  a=negative(a);b=negative(b);
  while(b!==0n){const t=a%b;a=b;b=t;}
  return a||1n;
}
export function rat(n,d=1n){
  if(typeof n!=='bigint'||typeof d!=='bigint')throw new TypeError('Exact rational numerators and denominators must be BigInt');
  if(d===0n)throw new RangeError('Exact rational zero denominator');
  if(n===0n)return ZERO;
  if(d<0n){n=-n;d=-d;}
  const g=gcd(n,d);
  return {n:n/g,d:d/g};
}
export const R0=ZERO;
export const R1=ONE;
export const rint=n=>rat(BigInt(n));
export const rneg=a=>({n:-a.n,d:a.d});
export const rabs=a=>a.n<0n?rneg(a):a;
export const radd=(a,b)=>rat(a.n*b.d+b.n*a.d,a.d*b.d);
export const rsub=(a,b)=>rat(a.n*b.d-b.n*a.d,a.d*b.d);
export const rmul=(a,b)=>rat(a.n*b.n,a.d*b.d);
export const rdiv=(a,b)=>rat(a.n*b.d,a.d*b.n);
export const rdivInt=(a,n)=>rat(a.n,a.d*BigInt(n));
export const rcmp=(a,b)=>a.n*b.d<b.n*a.d?-1:a.n*b.d>b.n*a.d?1:0;
export const req=(a,b)=>a.n===b.n&&a.d===b.d;
export const rkey=a=>a.n+'/'+a.d;

export function rfromNumber(value){
  if(!Number.isFinite(value))throw new RangeError('Cannot exactly rationalize non-finite trial measurement');
  if(Object.is(value,-0)||value===0)return ZERO;
  const view=new DataView(new ArrayBuffer(8));
  view.setFloat64(0,value,false);
  const bits=view.getBigUint64(0,false);
  const sign=(bits>>63n)!==0n?-1n:1n;
  const exponent=Number((bits>>52n)&0x7ffn);
  const fraction=bits&((1n<<52n)-1n);
  const mantissa=exponent===0?fraction:(1n<<52n)+fraction;
  const shift=exponent===0?-1074:exponent-1075;
  return shift>=0
    ?rat(sign*(mantissa<<BigInt(shift)))
    :rat(sign*mantissa,1n<<BigInt(-shift));
}

export function rmean(values){
  if(!values.length)throw new RangeError('Exact rational mean requires nonempty independent unit');
  let sum=ZERO;
  for(const value of values)sum=radd(sum,value);
  return rdivInt(sum,values.length);
}

export function rwire(value){
  const n=value.n.toString(),d=value.d.toString();
  let approximate;
  const direct=Number(value.n)/Number(value.d);
  if(Number.isFinite(direct)) approximate=direct;
  else {
    const neg=n[0]==='-',abs=neg?n.slice(1):n;
    const firstN=abs.slice(0,16),firstD=d.slice(0,16);
    const exponent=(abs.length-firstN.length)-(d.length-firstD.length);
    approximate=(neg?-1:1)*(Number(firstN)/Number(firstD))*Math.pow(10,exponent);
    if(!Number.isFinite(approximate))approximate=null;
  }
  return {numerator:n,denominator:d,approximate};
}

export function rwireToRat(wire){
  if(!wire||typeof wire.numerator!=='string'||typeof wire.denominator!=='string'
    ||!/^-?\d+$/.test(wire.numerator)||!/^[1-9]\d*$/.test(wire.denominator))
    throw new TypeError('Exact endpoint needs canonical integer numerator and positive denominator');
  return rat(BigInt(wire.numerator),BigInt(wire.denominator));
}
