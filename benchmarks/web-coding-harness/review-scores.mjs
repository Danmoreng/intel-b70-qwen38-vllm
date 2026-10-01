// Post-run public-contract review; no feedback to the timed agent.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const project=path.resolve(process.argv[2]);
const {createScoreStore}=await import(pathToFileURL(path.join(project,'src/scores.js')).href);
const checks=[];
function check(name,fn){try{fn();checks.push({name,passed:true});}catch(e){checks.push({name,passed:false,error:String(e)});}}
const entry={name:'Bench',score:3,frames:100,seed:12345};
check('storage read error reported by construction or actual access',()=>{
 assert.throws(()=>{const store=createScoreStore({getItem(){throw Error('unavailable');},setItem(){},removeItem(){}});store.list();});
});
check('storage write error reported by construction or actual record',()=>{
 assert.throws(()=>{const store=createScoreStore({getItem(){return null;},setItem(){throw Error('full');},removeItem(){throw Error('full');}});store.record(entry);});
});
check('public persisted schema loads and stays interoperable',()=>{
 let value=JSON.stringify({version:1,entries:[entry]});
 const storage={getItem(){return value;},setItem(_key,x){value=x;},removeItem(){value=null;}};
 const store=createScoreStore(storage);assert.deepEqual(store.list(),[entry]);
 const next={name:'Next',score:4,frames:90,seed:4242};store.record(next);
 const parsed=JSON.parse(value);assert.equal(parsed.version,1);assert.deepEqual(parsed.entries,[next,entry]);
});
console.log(JSON.stringify({purpose:'Supplemental contract review: eager and lazy storage adapters are both valid; unchanged timed outputs',checks,passed:checks.filter(x=>x.passed).length,total:checks.length}));
process.exitCode=checks.every(x=>x.passed)?0:1;
