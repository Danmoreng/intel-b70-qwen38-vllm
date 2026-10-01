# input public API contract

Export createInput(bindings={flap:['Space','ArrowUp'],pause:['KeyP'],restart:['KeyR']}), normalizePointer(event,rect,width,height), createTicker(step,options={hz:60,maxSteps:5}). Input object {handle,release,consume,reset,dispose}; handle({code,repeat?,type?}) uses type='keydown' default; keyup delegates release({code}). release accepts a key-event-like object with a code string, just like handle; recognized codes return true even when not held. It removes that key from the held set. Only known bindings, arrays of nonempty unique strings across actions; recognized nonrepeat keydown emits an edge action once while held, returns true; repeats/second held keydown recognized returns true without queueing; keyup returns recognized boolean. consume returns queued actions in order and empties; reset clears held and queue; dispose then future handles/releases return false and consume=[]; no global DOM listeners. normalizePointer maps finite clientX/clientY relative to finite rect {left,top,width,height} with positive dimensions to logical coordinates clamped0..width/height; logical width/height positive; malformed throws. Ticker returns {update,pause,resume,reset}; hz positive finite<=240, maxSteps integer1..100; update(timestampMs) first timestamp initializes, then computes elapsed fixed dt=1000/hz and invokes step(dt/1000) up to maxSteps; drops excess backlog, retains fractional remainder; decreasing timestamp rejects atomically; pause prevents callbacks and clears elapsed base; resume starts fresh base; reset fresh base. Reject nonfinite timestamps. No requestAnimationFrame internally.


Example of the recognized-key return contract:
```js
const input = createInput();
input.handle({code:'Space'}); // true, queues flap
input.handle({code:'Space'}); // true, does not queue another flap
input.consume(); // ['flap']
input.release({code:'Space'}); // true
input.handle({code:'Space',repeat:true}); // true, does not queue
input.consume(); // []
```
