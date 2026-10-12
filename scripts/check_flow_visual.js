// Local runtime check: traverse every stage and verify generated content/controls.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
class Element {
  constructor() { this.children=[]; this.listeners={}; this.textContent=''; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  createTHead() { const e=new Element(); this.append(e); return e; }
  createTBody() { const e=new Element(); this.append(e); return e; }
  insertRow() { const e=new Element(); this.append(e); return e; }
  insertCell() { const e=new Element(); this.append(e); return e; }
  addEventListener(name, callback) { this.listeners[name]=callback; }
}
const elements=Object.fromEntries(['previous','next','position','stage','content','detail'].map(name=>[name,new Element()]));
const root={querySelector(selector){const key=selector.slice(6,-1); assert(elements[key],selector); return elements[key];}};
const source=fs.readFileSync(process.argv[2],'utf8');
assert(!source.includes('__FLOW_DATA__'));
const script=source.match(/<script>([\s\S]*)<\/script>/)[1];
vm.runInNewContext(script,{document:{getElementById:id=>{assert.equal(id,'mage-flow-pilot-20261012');return root;},createElement:()=>new Element()},
  window:{addEventListener:()=>{},openai:{widgetState:{},setWidgetState:()=>Promise.resolve()}}});
const expected=['Observe','Embed entities','Attend to the board','Remember history','Infer the hidden hand','Score legal actions','Learn after the game'];
for(let i=0;i<expected.length;i++){
  assert.equal(elements.stage.textContent,expected[i]);
  assert(elements.content.children.length>0);
  if(i<6)elements.next.listeners.click();
}
assert(elements.next.disabled);elements.previous.listeners.click();assert.equal(elements.stage.textContent,expected[5]);
console.log('All seven flow stages and navigation passed the local DOM runtime check.');
