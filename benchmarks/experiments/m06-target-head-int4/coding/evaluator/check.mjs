import {spawnSync} from 'node:child_process';
process.chdir('/workspace/dashboard');
let failed=false;
for(const args of JSON.parse(process.argv[2])) {
  console.log('CHECK',JSON.stringify(args));
  const r=spawnSync(args[0],args.slice(1),{stdio:'inherit',env:{...process.env,CI:'1',B70_RELEASE_BUILD:'1',npm_config_cache:'/tmp/npm'}});
  if(r.status!==0){failed=true;console.log('CHECK_FAILED',r.status);if(process.argv[3]!=='continue')break;}
}
process.exitCode=failed?1:0;
