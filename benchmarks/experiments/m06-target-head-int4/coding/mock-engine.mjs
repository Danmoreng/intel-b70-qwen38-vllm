import {createServer} from 'node:http';
let n=0;
const values=()=>({'request_success_total':n,'request_prompt_tokens_sum':100*n,'request_generation_tokens_sum':10*n,'request_prefill_kv_computed_tokens_sum':100*n,'request_prefill_time_seconds_sum':n,'request_decode_time_seconds_sum':n,'prefix_cache_hits_total':0,'prefix_cache_queries_total':100*n,'spec_decode_num_accepted_tokens_total':4*n,'spec_decode_num_draft_tokens_total':8*n,'num_requests_running':0,'num_requests_waiting':0});
createServer(async(req,res)=>{
 if(req.url==='/metrics'){res.end(Object.entries(values()).map(([k,v])=>`vllm:${k} ${v}\n`).join(''));return;}
 let body='';for await(const c of req)body+=c;const d=JSON.parse(body);
 res.writeHead(200,{'content-type':'text/event-stream'});
 const delta=n===0?{role:'assistant',tool_calls:[{index:0,id:'read-smoke',type:'function',function:{name:'read',arguments:JSON.stringify({path:'/workspace/dashboard/package.json'})}}]}:{role:'assistant',content:'Sandbox protocol smoke complete.'};
 for(const e of [{id:'mock',object:'chat.completion.chunk',model:d.model,choices:[{index:0,delta,finish_reason:null}]},{id:'mock',object:'chat.completion.chunk',choices:[{index:0,delta:{},finish_reason:n===0?'tool_calls':'stop'}]},{id:'mock',object:'chat.completion.chunk',choices:[],usage:{prompt_tokens:100,completion_tokens:10,total_tokens:110}}])res.write('data: '+JSON.stringify(e)+'\n\n');
 res.end('data: [DONE]\n\n');setTimeout(()=>n++,200);
}).listen(18088,'127.0.0.1');
