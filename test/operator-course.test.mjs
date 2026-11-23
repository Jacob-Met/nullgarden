import test from 'node:test';
import assert from 'node:assert/strict';
import {buildOperatorCourse} from '../src/operator-course.mjs';

test('held inputs change independently and time shams preserve each spatial site',()=>{
  const source=buildOperatorCourse();
  assert.equal(source.schema,'nullgarden.operator-course/v1');
  for(const c of source.cases) {
    assert.ok(c.train.every(t=>t.spec.id!==c.held.spec.id));
    for(const t of c.train) {
      const width=c.kind==='periodic'?t.spec.initial.length:1;
      assert.equal(t.rows.length,width*t.states.length);
      for(let i=0;i<t.rows.length;i++){
        const j=i%width,time=Math.floor(i/width),target=(time+source.settings.permutationOffset)%t.states.length;
        const {change,...coordinates}=t.shuffled[i],{change:original,...originalCoordinates}=t.rows[i];
        assert.deepEqual(coordinates,originalCoordinates);
        assert.equal(change,t.rows[target*width+j].change);
      }
    }
  }
  assert.equal(source.cases[1].train[0].spec.initial.length,12);
  assert.equal(source.cases[1].held.spec.initial.length,15);
});

test('unobserved histories and spatial neighbours impose unavoidable residuals',()=>{
  const controls=buildOperatorCourse().outside;
  assert.deepEqual(controls.map(x=>x.id),['hidden_delay','hidden_lorenz_state','unseen_second_neighbour']);
  for(const [i,c] of controls.entries()){
    const coords=c.rows.map(({change,...rest})=>JSON.stringify(rest));
    assert.equal(new Set(coords).size,1);
    const y=c.rows.map(r=>r.change);
    assert.equal((Math.max(...y)-Math.min(...y))/2,[1,30,1][i]);
  }
});
