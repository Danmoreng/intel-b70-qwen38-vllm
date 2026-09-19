import assert from 'node:assert/strict';
import {mkdtempSync, writeFileSync, readFileSync, statSync, existsSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

process.chdir('/workspace/dashboard');
const results=[];
const base=`# operator comment retained\nOBSERVATORY_MODE=lan\nOBSERVATORY_HOST=192.168.1.9\nOBSERVATORY_PORT=8099\nOBSERVATORY_DB_PATH=/custom/state.db\nOBSERVATORY_WEB_ROOT=/custom/web\nOBSERVATORY_SESSION_DAYS=17\nOBSERVATORY_COOKIE_SECURE=true\nOBSERVATORY_PASSWORD_HASH=old-password-placeholder\nOBSERVATORY_GATEWAY_TOKEN_HASH=old-gateway-placeholder\nOBSERVATORY_AGENT_EXECUTION_MODE=isolated\nOBSERVATORY_AGENT_RUNNER_SOCKET=/custom/runner.sock\nOBSERVATORY_AGENT_RUNNER_TOKEN=${'a'.repeat(64)}\nOBSERVATORY_RELEASE_HELPER_SOCKET=/custom/release.sock\nOBSERVATORY_RELEASE_HELPER_TOKEN=${'b'.repeat(64)}\nCUSTOM_SETTING=keep-me\n`;
const parse=s=>Object.fromEntries(s.split('\n').filter(x=>x && !x.startsWith('#')).map(x=>{const i=x.indexOf('=');return[x.slice(0,i),x.slice(i+1)]}));
function cli(text,args=[],extra={}) {
  const dir=mkdtempSync('/tmp/rotation-eval-'), file=dir+'/observatory.env';
  if(text!==null)writeFileSync(file,text,{mode:0o600});
  const p=spawnSync('/usr/bin/node',['--require','/runtime/fake-terminal.cjs','packages/server/dist/cli/configure-auth.js','--output',file,...args],{encoding:'utf8',timeout:15000,env:{...process.env,...extra}});
  return {status:p.status,error:p.error?.message,output:p.stdout+p.stderr,text:existsSync(file)?readFileSync(file,'utf8'):null,mode:existsSync(file)?statSync(file).mode&0o777:null};
}
async function check(name,fn){try{await fn();results.push({name,pass:true})}catch(e){results.push({name,pass:false,error:String(e).slice(0,1600)})}}
await check('fresh setup and restrictive mode',()=>{
  const r=cli(null,['--host','127.0.0.1']);assert.equal(r.status,0,r.error??r.output);assert.equal(r.mode,0o600);assert.equal(parse(r.text).OBSERVATORY_HOST,'127.0.0.1');assert.ok(parse(r.text).OBSERVATORY_PASSWORD_HASH);assert.ok(parse(r.text).OBSERVATORY_GATEWAY_TOKEN_HASH);
});
await check('force rotation preserves all non-auth settings and comments',()=>{
  const r=cli(base,['--force']);assert.equal(r.status,0,r.output);assert.equal(r.mode,0o600);const before=parse(base), after=parse(r.text);
  for(const[k,v]of Object.entries(before))if(!['OBSERVATORY_PASSWORD_HASH','OBSERVATORY_GATEWAY_TOKEN_HASH'].includes(k))assert.equal(after[k],v,k);
  assert.ok(r.text.includes('# operator comment retained'));assert.notEqual(after.OBSERVATORY_PASSWORD_HASH,before.OBSERVATORY_PASSWORD_HASH);assert.notEqual(after.OBSERVATORY_GATEWAY_TOKEN_HASH,before.OBSERVATORY_GATEWAY_TOKEN_HASH);
  for(const secret of [before.OBSERVATORY_AGENT_RUNNER_TOKEN,before.OBSERVATORY_RELEASE_HELPER_TOKEN,before.OBSERVATORY_PASSWORD_HASH,before.OBSERVATORY_GATEWAY_TOKEN_HASH])assert.ok(!r.output.includes(secret),'existing secret leaked');
});
await check('force explicit host changes only host and auth values',()=>{
  const r=cli(base,['--force','--host','127.0.0.1']);assert.equal(r.status,0,r.output);const after=parse(r.text);assert.equal(after.OBSERVATORY_HOST,'127.0.0.1');
  for(const[k,v]of Object.entries(parse(base)))if(!['OBSERVATORY_HOST','OBSERVATORY_PASSWORD_HASH','OBSERVATORY_GATEWAY_TOKEN_HASH'].includes(k))assert.equal(after[k],v,k);
});
await check('password-only changes only password hash',()=>{
  const r=cli(base,['--password-only']);assert.equal(r.status,0,r.output);assert.equal(r.mode,0o600);assert.notEqual(parse(r.text).OBSERVATORY_PASSWORD_HASH,parse(base).OBSERVATORY_PASSWORD_HASH);assert.equal(r.text.replace(/^OBSERVATORY_PASSWORD_HASH=.*$/m,'PASSWORD'),base.replace(/^OBSERVATORY_PASSWORD_HASH=.*$/m,'PASSWORD'));
});
await check('mismatch preserves original bytes',()=>{const r=cli(base,['--force'],{EVAL_MISMATCH:'1'});assert.notEqual(r.status,0);assert.equal(r.text,base)});
await check('cancel preserves original bytes',()=>{const r=cli(base,['--force'],{EVAL_CANCEL:'1'});assert.notEqual(r.status,0);assert.equal(r.text,base)});
await check('duplicate setting is rejected without rewriting',()=>{const text=base+'OBSERVATORY_HOST=127.0.0.1\n';const r=cli(text,['--force']);assert.notEqual(r.status,0);assert.equal(r.text,text)});
await check('malformed setting is rejected without rewriting',()=>{const text=base+'NOT AN ENV ASSIGNMENT\n';const r=cli(text,['--force']);assert.notEqual(r.status,0);assert.equal(r.text,text)});
await check('disabled configuration and disconnected runner are distinct',async()=>{
  const {buildApp}=await import(pathToFileURL(process.cwd()+'/packages/server/dist/app.js'));
  const {loadServerConfig}=await import(pathToFileURL(process.cwd()+'/packages/server/dist/config.js'));
  const {RunnerAgentClient}=await import(pathToFileURL(process.cwd()+'/packages/server/dist/agent/runner-client.js'));
  const app=buildApp({version:'acceptance',config:loadServerConfig(),loggerLevel:'silent'});
  let response;try{response=await app.inject({method:'GET',url:'/api/agent'})}finally{await app.close()}
  assert.equal(response.statusCode,200);const disabled=response.json();assert.equal(disabled.enabled,false);
  assert.ok(!/until.*safety checks/i.test(disabled.reason??''),'unsupported safety-check claim');
  assert.ok(/config|configuration/i.test(JSON.stringify(disabled)),'configuration cause missing');
  const client=new RunnerAgentClient('/tmp/nonexistent-eval-runner.sock','a'.repeat(64));
  try{await new Promise(r=>setTimeout(r,100));const unavailable=client.snapshot();assert.equal(unavailable.enabled,false);assert.notEqual(unavailable.reason,disabled.reason);assert.match(unavailable.reason,/reach|connect|available/i)}finally{await client.close()}
});
console.log(JSON.stringify({schema:1,results,passed:results.filter(r=>r.pass).length,total:results.length},null,2));
process.exitCode=results.every(r=>r.pass)?0:1;
