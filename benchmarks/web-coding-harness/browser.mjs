import {chromium} from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
const project=path.resolve(process.argv[2]), stage=Number(process.argv[3]||6);
const screenshot=process.argv[4];
const rows=[], pageErrors=[];
const server=http.createServer(async(req,res)=>{
 try{
  const rel=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  if(rel==='/__harness__'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><html><head><meta charset="utf-8"></head><body><main id="root"></main></body></html>');return;}
  const file=path.resolve(project,'.'+(rel==='/'?'/index.html':rel));
  if(!file.startsWith(project+path.sep)){res.writeHead(403).end();return;}
  const types={'.js':'text/javascript','.html':'text/html','.css':'text/css','.json':'application/json'};
  res.setHeader('Content-Type',types[path.extname(file)]||'text/plain');res.end(await fs.readFile(file));
 }catch{res.writeHead(404).end('not found');}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const base='http://127.0.0.1:'+server.address().port;
let browser;
try{
 browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1000,height:900},deviceScaleFactor:1});
 page.on('pageerror',e=>pageErrors.push(String(e)));
 await page.goto(base+'/__harness__');
 const check=async(name,fn)=>{try{await fn();rows.push({name,passed:true});}catch(e){rows.push({name,passed:false,error:String(e).slice(0,2000)});}};
 const demand=(value,message)=>{if(!value)throw Error(message);};
 if(stage>=4){
 await check('WebGL2 procedural renderer and colors',async()=>{
  const x=await page.evaluate(async()=>{
   const {createRenderer}=await import('/src/renderer.js');const {createGame}=await import('/src/simulation.js');
   const c=document.createElement('canvas');document.body.append(c);window.__canvas=c;
   const r=createRenderer(c);window.__renderer=r;r.resize(480,720,1);const s=createGame();r.render(s);
   const gl=c.getContext('webgl2'),ext=gl.getExtension('WEBGL_debug_renderer_info');
   const pixels=new Uint8Array(480*720*4);gl.readPixels(0,0,480,720,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
   const colors=new Set();for(let i=0;i<pixels.length;i+=4)colors.add(pixels.slice(i,i+3).join(','));
   return {info:r.info(),colors:colors.size,gl:!!gl,gpu:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER),error:gl.getError()};
  });
  demand(x.gl&&x.info.backend==='webgl2','actual WebGL2 backend required');demand(x.colors>=4,'at least four drawn colors/regions required: '+JSON.stringify(x));demand(x.error===0,'WebGL errors');demand(/swiftshader/i.test(x.gpu),'software rendering required: '+x.gpu);rows.push({name:'browser renderer identity',passed:true,renderer:x.gpu});
 });
 await check('renderer DPR resize and state immutability',async()=>{
  const x=await page.evaluate(async()=>{const {createGame}=await import('/src/simulation.js');const s=createGame(),old=JSON.stringify(s);window.__renderer.resize(240,360,2);window.__renderer.render(s,{highContrast:true});return {w:window.__canvas.width,h:window.__canvas.height,css:window.__canvas.style.width,same:JSON.stringify(s)===old,info:window.__renderer.info()};});
  demand(x.w===480&&x.h===720&&x.same,'resize or state mutated');demand(x.info.dpr===2&&x.css==='240px','DPR/CSS contract');
 });
 await check('renderer context loss and restore',async()=>{
  await page.evaluate(()=>window.__canvas.getContext('webgl2').getExtension('WEBGL_lose_context').loseContext());
  await page.waitForFunction(()=>window.__renderer.info().contextLost===true,{},{timeout:3000});
  await page.evaluate(()=>window.__canvas.getContext('webgl2').getExtension('WEBGL_lose_context').restoreContext());
  await page.waitForFunction(()=>window.__renderer.info().contextLost===false,{},{timeout:5000});
  await page.evaluate(async()=>window.__renderer.render((await import('/src/simulation.js')).createGame()));
 });
 await check('renderer destroy idempotent and commands reject',async()=>{const x=await page.evaluate(()=>{const r=window.__renderer;r.destroy();r.destroy();let errors=0;try{r.resize(1,1);}catch{errors++;}try{r.render({});}catch{errors++;}return {errors,info:r.info()};});demand(x.errors===2&&x.info.destroyed,'destroy lifecycle');});
 }
 if(stage>=5){
 await page.goto(base+'/__harness__');
 await check('app mounts ready with accessible controls',async()=>{
  const x=await page.evaluate(async()=>{const {mountApp}=await import('/src/app.js');window.__app=mountApp(document.querySelector('#root'),{autoStart:false});const s=window.__app.getState();const ids=['game-canvas','start','pause','restart','score','status','live'];return {state:s,counts:ids.map(id=>document.querySelectorAll('[data-testid="'+id+'"]').length),live:document.querySelector('[data-testid="live"]')?.getAttribute('aria-live')};});
  demand(x.state.phase==='ready'&&x.counts.every(n=>n===1)&&x.live==='polite','app markup/ready state');
 });
 await check('app buttons start pause restart',async()=>{
  await page.click('[data-testid="start"]');demand(await page.evaluate(()=>window.__app.getState().phase==='running'),'start button');
  await page.evaluate(()=>window.__app.step(3));demand(await page.evaluate(()=>window.__app.getState().frame===3),'manual fixed steps');
  await page.click('[data-testid="pause"]');demand(await page.evaluate(()=>window.__app.getState().phase==='paused'),'pause button');
  await page.evaluate(()=>window.__app.step(2));demand(await page.evaluate(()=>window.__app.getState().frame===3),'paused must not tick');
  await page.click('[data-testid="pause"]');demand(await page.evaluate(()=>window.__app.getState().phase==='running'),'resume toggle');
  await page.click('[data-testid="restart"]');demand(await page.evaluate(()=>window.__app.getState().phase==='ready'&&window.__app.getState().frame===0),'restart reset');
 });
 await check('app keyboard and canvas pointer',async()=>{
  await page.keyboard.press('Space');demand(await page.evaluate(()=>window.__app.getState().phase==='running'),'Space starts');
  await page.keyboard.press('KeyP');demand(await page.evaluate(()=>window.__app.getState().phase==='paused'),'KeyP pauses');
  await page.keyboard.press('KeyR');await page.dispatchEvent('[data-testid="game-canvas"]','pointerdown',{clientX:100,clientY:100});demand(await page.evaluate(()=>window.__app.getState().phase==='running'),'pointer starts');
 });
 await check('app snapshot detached and invalid count',async()=>{const x=await page.evaluate(()=>{const s=window.__app.getState();s.bird.y=-900;let throws=0;for(const n of [-1,1.5,10001,NaN]){try{window.__app.step(n);}catch{throws++;}}return {same:window.__app.getState().bird.y!==-900,throws};});demand(x.same&&x.throws===4,'snapshot/count validation');});
 await check('app gameover and score/status text',async()=>{const x=await page.evaluate(()=>{window.__app.restart();window.__app.start();window.__app.step(1000);return {phase:window.__app.getState().phase,score:document.querySelector('[data-testid="score"]').textContent,status:document.querySelector('[data-testid="status"]').textContent};});demand(x.phase==='gameover'&&x.score.includes('0')&&x.status.toLowerCase().includes('game'),'gameover visible');});
 if(screenshot)await page.screenshot({path:screenshot,fullPage:true});
 await check('app destroy idempotent and removes input',async()=>{const x=await page.evaluate(()=>{window.__app.destroy();window.__app.destroy();let throws=0;for(const fn of ['start','step','flap','pause','resume','restart']){try{window.__app[fn]();}catch{throws++;}}return {throws,state:window.__app.getState()};});demand(x.throws===6,'commands after destroy');});
 if(stage>=6){await check('real entry point loads without JavaScript errors',async()=>{const old=pageErrors.length;await page.goto(base+'/');await page.waitForSelector('[data-testid="game-canvas"]',{timeout:3000});await page.waitForTimeout(300);demand(pageErrors.length===old,'entry errors '+pageErrors.slice(old).join(';'));});}
 }
 console.log(JSON.stringify({checks:rows,passed:rows.filter(x=>x.passed).length,total:rows.length,page_errors:pageErrors,browser_version:browser.version(),software_rendering:true}));
 process.exitCode=rows.every(x=>x.passed)?0:1;
}catch(e){console.log(JSON.stringify({harness_error:String(e.stack||e)}));process.exitCode=2;}
finally{await browser?.close();server.close();}
