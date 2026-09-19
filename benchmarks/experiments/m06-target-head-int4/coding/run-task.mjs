import {createServer,request as httpRequest} from 'node:http';
import {mkdirSync,writeFileSync,appendFileSync,readFileSync,chmodSync,rmSync,renameSync} from 'node:fs';
import {dirname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomBytes} from 'node:crypto';
import {createInterface} from 'node:readline';
import {StringDecoder} from 'node:string_decoder';
import {launchSandboxedJob,stopSandboxedJob} from './runtime/launcher.js';
const root=dirname(fileURLToPath(import.meta.url));
const [out,run,arm]=process.argv.slice(2);
const spec=JSON.parse(readFileSync(run+'/manifest.json'));
const endpoint=process.env.B70_BENCH_ROOT??'http://127.0.0.1:18087';
const id=randomBytes(12).toString('hex'),token=randomBytes(32).toString('hex');
const socketRoot='/tmp/b70-m06-coding-'+id;mkdirSync(socketRoot,{mode:0o700});
writeFileSync(out+'/sandbox-scope.json',JSON.stringify({runId:id}));
const profile=join(out,'profile');mkdirSync(profile,{recursive:true});
for(const f of ['models.json','settings.json'])writeFileSync(profile+'/'+f,readFileSync(root+'/profile/'+f));
let requests=[],busy=false,outputTokens=0,failure=null,job,turn=0,done=false,timer,lastAfter;
const tools=new Map();let toolTime=0;const start=performance.now();
const result={arm,status:'running',revision:spec.source_revision,requests,initial_and_followup_complete:false,compactions:[]};
function save(){writeFileSync(out+'/task-result.json.tmp',JSON.stringify({...result,failure_category:failure,wall_s:(performance.now()-start)/1000,tool_time_s:toolTime},null,2));renameSync(out+'/task-result.json.tmp',out+'/task-result.json');}
const metricNames={success:'vllm:request_success_total',prompt:'vllm:request_prompt_tokens_sum',generation:'vllm:request_generation_tokens_sum',computed:'vllm:request_prefill_kv_computed_tokens_sum',prefill:'vllm:request_prefill_time_seconds_sum',decode:'vllm:request_decode_time_seconds_sum',hits:'vllm:prefix_cache_hits_total',queries:'vllm:prefix_cache_queries_total',accepted:'vllm:spec_decode_num_accepted_tokens_total',drafted:'vllm:spec_decode_num_draft_tokens_total'};
async function metrics(){const response=await fetch(endpoint+'/metrics',{signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error('metrics HTTP '+response.status);const raw=await response.text(),vals={};for(const line of raw.split('\n')){const m=/^([^#{ ]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$/.exec(line);if(m)vals[m[1]]=(vals[m[1]]??0)+Number(m[2]);}for(const k of Object.values(metricNames))if(!Number.isFinite(vals[k]))throw Error('missing counter '+k);return{raw,vals};}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
const server=createServer(async(req,res)=>{
 if(req.method!=='POST'||req.url!=='/v1/chat/completions'||req.headers.authorization!=='Bearer '+token){res.writeHead(403);res.end();return;}
 if(busy||requests.length>=spec.limits.model_requests||outputTokens>=spec.limits.completion_tokens_total){failure='request_or_output_budget';res.writeHead(429,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:failure}}));return;}
 busy=true;let chunks=[],bytes=0;
 for await(const chunk of req){bytes+=chunk.length;if(bytes>16777216){req.destroy();busy=false;failure='body_limit';save();return;}chunks.push(chunk);}
 const n=requests.length,row={index:n,status:'running',phase:turn===0?'initial':'followup',started_s:(performance.now()-start)/1000};requests.push(row);save();
 let heldDone='';
 try{
  const body=JSON.parse(Buffer.concat(chunks).toString());
  body.max_tokens=Math.min(body.max_tokens??16384,16384,spec.limits.completion_tokens_total-outputTokens);
  body.thinking_token_budget=8192;body.temperature=1;body.top_p=.95;body.top_k=20;body.stream_options={...body.stream_options,include_usage:true};
  writeFileSync(`${out}/request-${n}.json`,JSON.stringify(body));
  const before=await metrics();writeFileSync(`${out}/metrics-${n}-before.txt`,before.raw);
  if(before.vals['vllm:num_requests_running']!==0||before.vals['vllm:num_requests_waiting']!==0)throw Error('concurrent request at boundary');
  if(lastAfter&&before.vals[metricNames.success]!==lastAfter[metricNames.success])throw Error('foreign request between turns');
  let stream='',pending='',firstByte=null,firstToken=null;const decoder=new StringDecoder('utf8');const sent=performance.now();
  function forwardFrames(){let at;while((at=pending.indexOf('\n\n'))>=0){const frame=pending.slice(0,at+2);pending=pending.slice(at+2);if(frame.includes('data: [DONE]')){heldDone+=frame;continue;}for(const line of frame.split('\n')){if(!line.startsWith('data: '))continue;try{const ev=JSON.parse(line.slice(6));if(ev.usage)row.usage=ev.usage;for(const ch of ev.choices??[]){if(ch.finish_reason)row.finish_reason=ch.finish_reason;const d=ch.delta??{};if(firstToken===null&&(d.content||d.reasoning||d.reasoning_content||d.tool_calls?.length))firstToken=performance.now();}}catch{}}res.write(frame);}}
  await new Promise((resolve,reject)=>{
   const upstream=httpRequest(endpoint+'/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'}},response=>{
    row.http_status=response.statusCode;res.writeHead(response.statusCode??502,{'content-type':response.headers['content-type']??'text/event-stream'});
    response.on('data',chunk=>{if(firstByte===null)firstByte=performance.now();const text=decoder.write(chunk);stream+=text;pending+=text;if(stream.length>67108864){upstream.destroy(new Error('response limit'));return;}forwardFrames();});
    response.on('end',resolve);response.on('error',reject);
   });upstream.on('error',reject);upstream.setTimeout(600000,()=>upstream.destroy(new Error('upstream timeout')));upstream.end(JSON.stringify(body));
  });
  const tail=decoder.end();stream+=tail;pending+=tail;forwardFrames();heldDone+=pending;
  writeFileSync(`${out}/response-${n}.sse`,stream);
  row.wall_s=(performance.now()-sent)/1000;row.first_sse_byte_s=firstByte===null?null:(firstByte-sent)/1000;row.ttft_s=firstToken===null?null:(firstToken-sent)/1000;
  outputTokens+=row.usage?.completion_tokens??0;
  let after;const accountingStart=performance.now();
  for(let retry=0;retry<40;retry++){after=await metrics();if(after.vals[metricNames.success]-before.vals[metricNames.success]>=1)break;await pause(100);}
  row.accounting_wait_s=(performance.now()-accountingStart)/1000;
  writeFileSync(`${out}/metrics-${n}-after.txt`,after.raw);lastAfter=after.vals;
  row.metric_deltas={};for(const[k,v]of Object.entries(after.vals))if(/(_sum|_count|_total)$/.test(k)&&k in before.vals)row.metric_deltas[k]=v-before.vals[k];
  const d=row.metric_deltas;
  if(Object.entries(d).some(([k,v])=>v<0&&k.startsWith('vllm:')))throw Error('metrics reset');
  if(d[metricNames.success]!==1)throw Error('completion counter mismatch');
  if(!row.usage||d[metricNames.prompt]!==row.usage.prompt_tokens||d[metricNames.generation]!==row.usage.completion_tokens)throw Error('usage/counter mismatch');
  row.native_prefill_s=d[metricNames.prefill];row.native_decode_s=d[metricNames.decode];
  row.prompt_tokens=d[metricNames.prompt];row.generated_tokens=d[metricNames.generation];row.computed_tokens=d[metricNames.computed];row.prefix_hit_tokens=d[metricNames.hits];row.prefix_query_tokens=d[metricNames.queries];row.accepted_tokens=d[metricNames.accepted];row.drafted_tokens=d[metricNames.drafted];
  row.status=row.http_status===200&&['stop','tool_calls','length'].includes(row.finish_reason)?'ok':'inference_error';
  if(row.status!=='ok')failure='inference_error';
 }catch(e){row.status='infrastructure_error';row.error=String(e);failure='infrastructure_error';if(!res.headersSent)res.writeHead(502);}
 busy=false;save();res.end(heldDone);
});
await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(socketRoot+'/relay.sock',resolve)});chmodSync(socketRoot+'/relay.sock',0o666);
function send(message){job.child.stdin.write(JSON.stringify({type:'prompt',message})+'\n');}
async function finish(status){if(done)return;done=true;clearTimeout(timer);result.status=status;result.initial_and_followup_complete=turn===2&&!failure;save();if(job)await stopSandboxedJob(id).catch(()=>{});server.closeAllConnections();server.close();rmSync(socketRoot,{recursive:true,force:true});process.exit(status==='completed'?0:1);}
job=launchSandboxedJob({runId:id,workspace:{path:join(out,'workspace'),revision:spec.source_revision,runId:id,branch:`agent/${id}`},dependencyRoot:run+'/dependencies',dependencyTarget:'dashboard/node_modules',runtimeRoot:root+'/runtime',runtimeExecutable:'sandbox-entry.js',runtimeNodeEntrypoint:true,profileRoot:profile,writableProfile:true,agentRuntimeRoot:spec.agent_runtime,agentExecutable:'@earendil-works/pi-coding-agent/dist/bundle/cli.js',inferenceSocketRoot:socketRoot,inferenceToken:token,args:['--mode','rpc','--no-session','--offline','--provider','b70-vllm','--model','Qwen3.8-27B','--thinking','medium','--append-system-prompt','The selected project is /workspace/dashboard. Work only there. All builds and tests must run with cwd /workspace/dashboard. Do not install dependencies or use the network. Dependencies are preinstalled. Use B70_RELEASE_BUILD=1 npm test to omit host-only nested sandbox checks.'],limits:{memoryBytes:4294967296,cpuQuotaPercent:200,tasksMax:128,runtimeSeconds:spec.limits.wall_seconds+30,tmpBytes:536870912}});
job.child.stderr.on('data',c=>appendFileSync(out+'/pi-stderr.log',c));
createInterface({input:job.child.stdout}).on('line',line=>{appendFileSync(out+'/pi-events.jsonl',JSON.stringify({at_s:(performance.now()-start)/1000,event:line})+'\n');let ev;try{ev=JSON.parse(line)}catch{return;}
 if(ev.type==='tool_execution_start')tools.set(ev.toolCallId,performance.now());
 if(ev.type==='tool_execution_end'&&tools.has(ev.toolCallId)){toolTime+=(performance.now()-tools.get(ev.toolCallId))/1000;tools.delete(ev.toolCallId);}
 if(ev.type.includes('compaction')){result.compactions.push({type:ev.type,at_s:(performance.now()-start)/1000,request_index:requests.length});save();}
 if(ev.type==='message_end'&&ev.message?.role==='assistant'&&['error','aborted'].includes(ev.message.stopReason))failure??='model_error';
 if(ev.type==='agent_settled'){turn++;if(failure)void finish('failed');else if(turn===1)send(readFileSync(root+'/followup.md','utf8'));else void finish('completed');}
});
job.child.on('error',e=>{failure=String(e);void finish('infrastructure_error')});job.child.on('close',()=>{if(!done){failure??='unexpected_pi_exit';void finish('failed')}});
timer=setTimeout(()=>{failure='task_timeout';void finish('failed')},spec.limits.wall_seconds*1000);
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{failure='interrupted';void finish('interrupted')});
send(readFileSync(root+'/task.md','utf8'));save();
